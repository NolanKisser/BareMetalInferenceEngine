#include "tokenizer.h"

#include <fstream>
#include <iostream>
#include <queue>

namespace {

// U+2581, which SentencePiece uses in place of a space.
const std::string kSpaceMarker = "\xE2\x96\x81";
const std::string kReplacementChar = "\xEF\xBF\xBD";

bool fail(const std::string& message) {
    std::cerr << "Cannot load tokenizer: " << message << std::endl;
    return false;
}

int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// The export writes each string as hex, or "-" when it is empty, so strings with
// spaces or newlines fit on one line.
bool parse_hex(const std::string& hex, std::string& out) {
    out.clear();
    if (hex == "-") return true;
    if (hex.empty() || hex.size() % 2 != 0) return false;
    for (size_t i = 0; i < hex.size(); i += 2) {
        const int high = hex_digit(hex[i]), low = hex_digit(hex[i + 1]);
        if (high < 0 || low < 0) return false;
        out += static_cast<char>(high * 16 + low);
    }
    return true;
}

// Length of the UTF-8 character starting at text[pos]. A byte that does not
// start a valid character counts as one character on its own.
size_t utf8_length(const std::string& text, size_t pos) {
    const unsigned char lead = static_cast<unsigned char>(text[pos]);
    size_t length = 1;
    if (lead >= 0xC2 && lead <= 0xDF) length = 2;
    else if (lead >= 0xE0 && lead <= 0xEF) length = 3;
    else if (lead >= 0xF0 && lead <= 0xF4) length = 4;
    if (pos + length > text.size()) return 1;
    for (size_t i = 1; i < length; ++i) {
        if ((static_cast<unsigned char>(text[pos + i]) & 0xC0) != 0x80) return 1;
    }
    return length;
}

bool valid_utf8(const std::string& text) {
    for (size_t pos = 0; pos < text.size();) {
        const unsigned char lead = static_cast<unsigned char>(text[pos]);
        const size_t length = utf8_length(text, pos);
        if (length == 1 && lead >= 0x80) return false;
        if (length == 3) {
            const unsigned char second = static_cast<unsigned char>(text[pos + 1]);
            if ((lead == 0xE0 && second < 0xA0) || (lead == 0xED && second > 0x9F)) return false;  // overlong, surrogate
        } else if (length == 4) {
            const unsigned char second = static_cast<unsigned char>(text[pos + 1]);
            if ((lead == 0xF0 && second < 0x90) || (lead == 0xF4 && second > 0x8F)) return false;  // overlong, past U+10FFFF
        }
        pos += length;
    }
    return true;
}

}  // namespace

bool Tokenizer::load(const std::string& path) {
    *this = Tokenizer{};
    std::ifstream file(path);
    if (!file) return fail("could not open " + path);

    std::string format;
    int version = 0;
    if (!(file >> format >> version) || format != "baremetal-tokenizer" || version != 1) {
        return fail(path + " is not a version 1 tokenizer");
    }

    Tokenizer parsed;
    long long vocab_count = -1;
    std::string key;
    while (vocab_count < 0 && file >> key) {
        bool ok = true;
        int flag = 0;
        if (key == "bos") ok = static_cast<bool>(file >> parsed.bos_id_);
        else if (key == "eos") ok = static_cast<bool>(file >> parsed.eos_id_);
        else if (key == "unk") ok = static_cast<bool>(file >> parsed.unk_id_);
        else if (key == "add_bos") { ok = (file >> flag) && (flag == 0 || flag == 1); parsed.add_bos_ = flag == 1; }
        else if (key == "byte_fallback") { ok = (file >> flag) && (flag == 0 || flag == 1); parsed.byte_fallback_ = flag == 1; }
        else if (key == "vocab") ok = (file >> vocab_count) && vocab_count > 0;
        else return fail("unknown setting '" + key + "' in " + path);
        if (!ok) return fail("invalid value for '" + key + "' in " + path);
    }
    if (vocab_count < 0) return fail(path + " has no vocabulary");

    parsed.byte_ids_.assign(256, -1);
    parsed.byte_values_.assign(static_cast<size_t>(vocab_count), -1);
    for (long long id = 0; id < vocab_count; ++id) {
        std::string hex, piece;
        if (!(file >> hex) || !parse_hex(hex, piece)) return fail(path + " has an invalid or missing token " + std::to_string(id));
        // Added tokens may repeat a piece. The first id wins, as in Hugging Face.
        parsed.ids_.emplace(piece, static_cast<int>(id));
        if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece[5] == '>') {
            const int high = hex_digit(piece[3]), low = hex_digit(piece[4]);
            if (high >= 0 && low >= 0) {
                const int value = high * 16 + low;
                parsed.byte_values_[id] = value;
                if (parsed.byte_ids_[value] < 0) parsed.byte_ids_[value] = static_cast<int>(id);
            }
        }
        parsed.pieces_.push_back(std::move(piece));
    }

    auto in_vocab = [&](int id) { return id >= 0 && id < static_cast<int>(parsed.pieces_.size()); };
    if (parsed.add_bos_ && !in_vocab(parsed.bos_id_)) return fail(path + " adds a start token that is not in the vocabulary");
    if (parsed.unk_id_ >= 0 && !in_vocab(parsed.unk_id_)) return fail(path + " has an unknown-token id outside the vocabulary");

    parsed.special_.assign(parsed.pieces_.size(), false);
    parsed.decode_pieces_ = parsed.pieces_;
    long long added_count = -1;
    if (!(file >> key >> added_count) || key != "added" || added_count < 0) return fail(path + " does not list its added tokens");
    for (long long i = 0; i < added_count; ++i) {
        AddedToken token;
        int special = -1;
        std::string hex;
        if (!(file >> token.id >> special >> hex) || !in_vocab(token.id) || (special != 0 && special != 1) ||
            !parse_hex(hex, token.match) || token.match.empty()) {
            return fail(path + " has an invalid added token");
        }
        token.special = special == 1;
        parsed.special_[token.id] = token.special;
        // Hugging Face decodes an added token as its normalized text, so </s>
        // comes out as " </s>". Encoding still uses the original piece.
        parsed.decode_pieces_[token.id] = token.match;
        parsed.added_.push_back(std::move(token));
    }

    long long merge_count = -1;
    if (!(file >> key >> merge_count) || key != "merges" || merge_count < 0) return fail(path + " does not list its merges");
    parsed.merges_.reserve(static_cast<size_t>(merge_count));
    for (long long rank = 0; rank < merge_count; ++rank) {
        std::string left_hex, right_hex, left, right;
        if (!(file >> left_hex >> right_hex) || !parse_hex(left_hex, left) || !parse_hex(right_hex, right)) {
            return fail(path + " has an invalid merge at rank " + std::to_string(rank));
        }
        const auto l = parsed.ids_.find(left), r = parsed.ids_.find(right), merged = parsed.ids_.find(left + right);
        if (l == parsed.ids_.end() || r == parsed.ids_.end() || merged == parsed.ids_.end()) {
            return fail("merge " + std::to_string(rank) + " in " + path + " uses a token missing from the vocabulary");
        }
        // A repeated pair keeps its first, highest-priority rank.
        parsed.merges_.emplace(pair_key(l->second, r->second), Merge{static_cast<int>(rank), merged->second});
    }
    if (file >> key) return fail(path + " has unexpected data after its merges");

    *this = std::move(parsed);
    return true;
}

bool Tokenizer::is_special(int id) const {
    return id >= 0 && id < static_cast<int>(special_.size()) && special_[id];
}

bool Tokenizer::is_byte(int id) const {
    return id >= 0 && id < static_cast<int>(byte_values_.size()) && byte_values_[id] >= 0;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> out;
    if (add_bos && add_bos_) out.push_back(bos_id_);
    if (text.empty()) return out;

    std::string normalized = kSpaceMarker;
    for (char c : text) {
        if (c == ' ') normalized += kSpaceMarker;
        else normalized += c;
    }

    // Split out added tokens, taking the longest match at the leftmost position.
    // Matching happens after normalization, so "</s>" only matches after a space.
    size_t piece_start = 0;
    for (size_t pos = 0; pos < normalized.size();) {
        const AddedToken* best = nullptr;
        for (const AddedToken& token : added_) {
            if (normalized.compare(pos, token.match.size(), token.match) == 0 &&
                (!best || token.match.size() > best->match.size())) {
                best = &token;
            }
        }
        if (!best) {
            ++pos;
            continue;
        }
        encode_piece(normalized.substr(piece_start, pos - piece_start), out);
        out.push_back(best->id);
        pos += best->match.size();
        piece_start = pos;
    }
    encode_piece(normalized.substr(piece_start), out);
    return out;
}

// Follows Word::merge_all in Hugging Face tokenizers: repeatedly merges the pair
// with the lowest rank, taking the leftmost one on a tie.
void Tokenizer::encode_piece(const std::string& piece, std::vector<int>& out) const {
    if (piece.empty()) return;

    struct Symbol {
        int id;
        int prev;
        int next;
        bool alive;
    };
    std::vector<Symbol> symbols;
    bool last_was_unknown = false;
    auto push = [&](int id) {
        const int index = static_cast<int>(symbols.size());
        symbols.push_back({id, index - 1, -1, true});
        if (index > 0) symbols[index - 1].next = index;
    };

    for (size_t pos = 0; pos < piece.size();) {
        const size_t length = utf8_length(piece, pos);
        const auto found = ids_.find(piece.substr(pos, length));
        if (found != ids_.end()) {
            push(found->second);
            last_was_unknown = false;
        } else {
            bool all_bytes = byte_fallback_;
            for (size_t i = 0; all_bytes && i < length; ++i) {
                all_bytes = byte_ids_[static_cast<unsigned char>(piece[pos + i])] >= 0;
            }
            if (all_bytes) {
                for (size_t i = 0; i < length; ++i) push(byte_ids_[static_cast<unsigned char>(piece[pos + i])]);
                last_was_unknown = false;
            } else if (unk_id_ >= 0) {
                // Consecutive unknown characters become one unknown token.
                if (!last_was_unknown) push(unk_id_);
                last_was_unknown = true;
            }
        }
        pos += length;
    }

    struct Candidate {
        int rank;
        int pos;
        int new_id;
        bool operator<(const Candidate& other) const {
            // priority_queue pops the largest, so invert: lowest rank, then leftmost.
            if (rank != other.rank) return rank > other.rank;
            return pos > other.pos;
        }
    };
    std::priority_queue<Candidate> queue;
    auto consider = [&](int left) {
        const int right = symbols[left].next;
        if (right < 0) return;
        const auto merge = merges_.find(pair_key(symbols[left].id, symbols[right].id));
        if (merge != merges_.end()) queue.push({merge->second.rank, left, merge->second.new_id});
    };
    for (int i = 0; i + 1 < static_cast<int>(symbols.size()); ++i) consider(i);

    while (!queue.empty()) {
        const Candidate top = queue.top();
        queue.pop();
        Symbol& left = symbols[top.pos];
        if (!left.alive || left.next < 0) continue;
        // Skip candidates made stale by an earlier merge on either side.
        const auto merge = merges_.find(pair_key(left.id, symbols[left.next].id));
        if (merge == merges_.end() || merge->second.new_id != top.new_id) continue;

        Symbol& right = symbols[left.next];
        right.alive = false;
        left.id = top.new_id;
        left.next = right.next;
        if (left.next >= 0) symbols[left.next].prev = top.pos;
        if (left.prev >= 0) consider(left.prev);
        consider(top.pos);
    }

    for (const Symbol& symbol : symbols) {
        if (symbol.alive) out.push_back(symbol.id);
    }
}

// Follows the decoder chain in Llama's tokenizer.json: U+2581 becomes a space,
// <0xNN> tokens become bytes, and one leading space is removed.
std::string Tokenizer::decode(const std::vector<int>& tokens, bool skip_special) const {
    std::string text, bytes;
    int byte_tokens = 0;
    auto flush_bytes = [&]() {
        if (byte_tokens == 0) return;
        if (valid_utf8(bytes)) {
            text += bytes;
        } else {
            for (int i = 0; i < byte_tokens; ++i) text += kReplacementChar;
        }
        bytes.clear();
        byte_tokens = 0;
    };

    for (int id : tokens) {
        if (id < 0 || id >= vocab_size()) continue;
        if (skip_special && special_[id]) continue;
        if (byte_values_[id] >= 0) {
            bytes += static_cast<char>(byte_values_[id]);
            ++byte_tokens;
            continue;
        }
        flush_bytes();
        const std::string& piece = decode_pieces_[id];
        for (size_t pos = 0; pos < piece.size();) {
            if (piece.compare(pos, kSpaceMarker.size(), kSpaceMarker) == 0) {
                text += ' ';
                pos += kSpaceMarker.size();
            } else {
                text += piece[pos++];
            }
        }
    }
    flush_bytes();
    if (!text.empty() && text[0] == ' ') text.erase(0, 1);
    return text;
}
