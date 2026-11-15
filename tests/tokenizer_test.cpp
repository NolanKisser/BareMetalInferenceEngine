// CPU-only checks, so they can run on CI machines that have no GPU.
//
// Usage: tokenizer_test [export_dir]
// Without arguments, checks a small tokenizer written by this test. With an
// export directory, also compares tokenizer.txt with tokenizer_reference.txt,
// which scripts/reference_tokens.py writes from Hugging Face tokenizers.
#include "tokenizer.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool passed, const std::string& name) {
    if (!passed) ++failures;
    std::printf("%s  %s\n", passed ? "PASS" : "FAIL", name.c_str());
}

std::string to_hex(const std::string& text) {
    if (text.empty()) return "-";
    static const char* digits = "0123456789abcdef";
    std::string out;
    for (unsigned char c : text) {
        out += digits[c >> 4];
        out += digits[c & 15];
    }
    return out;
}

std::string from_hex(const std::string& hex) {
    std::string out;
    if (hex == "-") return out;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) out += static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16));
    return out;
}

std::string join(const std::vector<int>& ids) {
    std::string out;
    for (int id : ids) out += (out.empty() ? "" : " ") + std::to_string(id);
    return out;
}

void write_file(const std::string& path, const std::string& contents) {
    std::ofstream file(path, std::ios::binary);
    file << contents;
}

const std::string kMarker = "\xE2\x96\x81";  // U+2581
const std::string kReplacement = "\xEF\xBF\xBD";

// A Llama-style layout: <unk>, <s>, </s>, then <0x00> to <0xFF> as ids 3 to 258.
std::string small_tokenizer(const std::string& extra_merge = "") {
    std::ostringstream out;
    out << "baremetal-tokenizer 1\nbos 1\neos 2\nunk 0\nadd_bos 1\nbyte_fallback 1\n";
    std::vector<std::string> pieces = {"<unk>", "<s>", "</s>"};
    for (int b = 0; b < 256; ++b) {
        char name[8];
        std::snprintf(name, sizeof(name), "<0x%02X>", b);
        pieces.push_back(name);
    }
    // ids 259 onward
    for (const std::string& piece : {kMarker, std::string("a"), std::string("b"), std::string("c"), kMarker + "a",
                                     std::string("ab"), std::string("bc"), kMarker + "ab", std::string("abc"),
                                     std::string("\xC3\xA9")}) {
        pieces.push_back(piece);
    }
    out << "vocab " << pieces.size() << "\n";
    for (const std::string& piece : pieces) out << to_hex(piece) << "\n";
    out << "added 3\n0 1 " << to_hex(kMarker + "<unk>") << "\n1 1 " << to_hex(kMarker + "<s>") << "\n2 1 "
        << to_hex(kMarker + "</s>") << "\n";
    // Ranks: "b c" outranks "a b", so "abc" does not merge left to right.
    const std::vector<std::pair<std::string, std::string>> merges = {
        {kMarker, "a"}, {"b", "c"}, {"a", "b"}, {kMarker + "a", "b"}, {"ab", "c"}};
    out << "merges " << merges.size() + (extra_merge.empty() ? 0 : 1) << "\n";
    for (const auto& merge : merges) out << to_hex(merge.first) << " " << to_hex(merge.second) << "\n";
    if (!extra_merge.empty()) out << extra_merge << "\n";
    return out.str();
}

int byte_token(int value) { return 3 + value; }

void check_small_tokenizer() {
    Tokenizer tokenizer;
    check(!tokenizer.load("this_tokenizer_does_not_exist.txt"), "load reports a missing file");

    const std::string path = "tokenizer_test_small.txt";
    write_file(path, "not-a-tokenizer 1\n");
    check(!tokenizer.load(path), "load rejects a file with the wrong header");
    write_file(path, small_tokenizer(to_hex("a") + " " + to_hex("z")));
    check(!tokenizer.load(path), "load rejects a merge that uses an unknown token");
    write_file(path, small_tokenizer() + "extra\n");
    check(!tokenizer.load(path), "load rejects data after the merges");

    write_file(path, small_tokenizer());
    const bool loaded = tokenizer.load(path);
    std::remove(path.c_str());
    check(loaded, "load reads a valid file");
    if (!loaded) return;
    check(tokenizer.vocab_size() == 269 && tokenizer.bos_id() == 1 && tokenizer.eos_id() == 2, "load reads the settings");

    const int marker = 259, a = 260, marker_a = 263, bc = 265, marker_ab = 266, e_acute = 268;
    check(tokenizer.encode("") == std::vector<int>{1}, "empty text encodes to the start token");
    check(tokenizer.encode("", false).empty(), "empty text without the start token encodes to nothing");
    check(tokenizer.encode("ab") == std::vector<int>{1, marker_ab}, "merges apply repeatedly");
    check(tokenizer.encode("abc") == std::vector<int>{1, marker_a, bc}, "the lowest-ranked pair merges first");
    check(tokenizer.encode("a", false) == std::vector<int>{marker_a}, "add_bos false leaves out the start token");
    check(tokenizer.encode("\xC3\xA9") == std::vector<int>{1, marker, e_acute}, "a multi-byte character in the vocabulary is one token");
    check(tokenizer.encode("\xC3\xBC") == std::vector<int>{1, marker, byte_token(0xC3), byte_token(0xBC)},
          "a character missing from the vocabulary falls back to bytes");
    check(tokenizer.encode("a </s>") == std::vector<int>{1, marker_a, 2}, "an added token after a space is matched");
    const std::vector<int> unmatched = tokenizer.encode("a</s>");
    bool has_eos = false;
    for (int id : unmatched) has_eos = has_eos || id == 2;
    check(!has_eos, "an added token with no space before it is not matched, as in Hugging Face");

    check(tokenizer.decode({1, marker_a, bc}) == "abc", "decode drops added tokens and the leading space");
    check(tokenizer.decode({marker_a, 2, marker_a}, false) == "a </s> a", "decode can keep added tokens, with a space before each");
    check(tokenizer.decode({byte_token(0xC3), byte_token(0xBC)}) == "\xC3\xBC", "byte tokens decode to UTF-8");
    check(tokenizer.decode({byte_token(0xC3)}) == kReplacement, "an incomplete character decodes to U+FFFD");
    check(tokenizer.decode({byte_token(0xC3), a, byte_token(0xBC)}) == kReplacement + "a" + kReplacement,
          "each invalid byte becomes its own U+FFFD");
    check(tokenizer.is_byte(byte_token(0x41)) && !tokenizer.is_byte(a), "is_byte identifies byte tokens");
    check(tokenizer.is_special(2) && !tokenizer.is_special(a), "is_special identifies added tokens");

    for (const std::string text : {"abc cab", "  a  b  ", "caf\xC3\xA9 \xF0\x9F\x9A\x80", "line\nbreak", ""}) {
        check(tokenizer.decode(tokenizer.encode(text)) == text, "\"" + text + "\" survives encode then decode");
    }
}

void check_reference(const std::string& dir) {
    Tokenizer tokenizer;
    if (!tokenizer.load(dir + "/tokenizer.txt")) {
        check(false, "load the exported tokenizer");
        return;
    }
    std::ifstream file(dir + "/tokenizer_reference.txt");
    std::string header;
    if (!std::getline(file, header) || header != "baremetal-tokenizer-reference 1") {
        check(false, "read " + dir + "/tokenizer_reference.txt");
        return;
    }

    int encodes = 0, decodes = 0, encode_failures = 0, decode_failures = 0;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream fields(line);
        std::string kind, hex;
        int count = 0;
        fields >> kind;
        if (kind == "encode") {
            fields >> hex >> count;
            std::vector<int> expected(count);
            for (int& id : expected) fields >> id;
            const std::string text = from_hex(hex);
            const std::vector<int> actual = tokenizer.encode(text);
            ++encodes;
            if (actual != expected && ++encode_failures <= 5) {
                std::printf("  encode %s\n    expected %s\n    actual   %s\n", hex.c_str(), join(expected).c_str(), join(actual).c_str());
            }
        } else if (kind == "decode") {
            fields >> count;
            std::vector<int> ids(count);
            for (int& id : ids) fields >> id;
            std::string with_added, without_added;
            fields >> with_added >> without_added;
            ++decodes;
            const std::string actual_with = to_hex(tokenizer.decode(ids, false));
            const std::string actual_without = to_hex(tokenizer.decode(ids, true));
            if ((actual_with != with_added || actual_without != without_added) && ++decode_failures <= 5) {
                std::printf("  decode %s\n    expected %s / %s\n    actual   %s / %s\n", join(ids).c_str(), with_added.c_str(),
                            without_added.c_str(), actual_with.c_str(), actual_without.c_str());
            }
        }
    }
    check(encodes > 0 && encode_failures == 0,
          std::to_string(encodes - encode_failures) + " of " + std::to_string(encodes) + " encodings match Hugging Face");
    check(decodes > 0 && decode_failures == 0,
          std::to_string(decodes - decode_failures) + " of " + std::to_string(decodes) + " decodings match Hugging Face");
}

}  // namespace

int main(int argc, char** argv) {
    check_small_tokenizer();
    if (argc > 1) check_reference(argv[1]);

    std::printf("\n%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
