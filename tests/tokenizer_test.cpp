// CPU-only checks, so they can run on CI machines that have no GPU.
#include "tokenizer.h"

#include <cstdio>
#include <fstream>
#include <string>

namespace {

int failures = 0;

void check(bool passed, const char* name) {
    if (!passed) ++failures;
    std::printf("%s  %s\n", passed ? "PASS" : "FAIL", name);
}

}  // namespace

int main() {
    Tokenizer tokenizer;
    check(!tokenizer.load("this_vocab_file_does_not_exist.txt"), "load reports a missing vocab file");

    const std::string vocab_path = "tokenizer_test_vocab.txt";
    {
        std::ofstream vocab(vocab_path);
        vocab << "A\nB\nC\n";
    }
    check(tokenizer.load(vocab_path), "load reads an existing vocab file");
    std::remove(vocab_path.c_str());

    for (const std::string text : {"Hello, BareMetal!", "", "caf\xC3\xA9 \xF0\x9F\x9A\x80"}) {
        check(tokenizer.decode(tokenizer.encode(text)) == text, "encode then decode returns the original text");
    }

    std::printf("\n%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
