#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Byte-pair encoding for SentencePiece-style Llama tokenizers, matching the
// Hugging Face `tokenizers` library. It reads the tokenizer.txt file that
// scripts/export_llama.py writes from tokenizer.json.
//
// Encoding replaces spaces with U+2581 and adds one at the start, splits out
// added tokens such as </s>, then merges each remaining piece. A character
// missing from the vocabulary becomes one <0xNN> token per UTF-8 byte.
class Tokenizer {
public:
    // Returns false, after printing the reason, if the file is missing or invalid.
    bool load(const std::string& path);

    std::vector<int> encode(const std::string& text, bool add_bos = true) const;

    // Turns <0xNN> tokens back into bytes. Bytes that do not form valid UTF-8
    // become U+FFFD, one per byte. Drops added tokens when skip_special is set.
    std::string decode(const std::vector<int>& tokens, bool skip_special = true) const;

    int vocab_size() const { return static_cast<int>(pieces_.size()); }
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    bool is_special(int id) const;
    // True for a <0xNN> token, which may be part of a multi-byte character.
    bool is_byte(int id) const;

private:
    struct Merge {
        int rank;
        int new_id;
    };
    struct AddedToken {
        int id;
        std::string match;  // the token's text after normalization
        bool special;
    };

    void encode_piece(const std::string& piece, std::vector<int>& out) const;
    static uint64_t pair_key(int left, int right) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) | static_cast<uint32_t>(right);
    }

    std::vector<std::string> pieces_;
    std::vector<std::string> decode_pieces_;  // pieces_, with added tokens normalized
    std::unordered_map<std::string, int> ids_;
    std::unordered_map<uint64_t, Merge> merges_;
    std::vector<AddedToken> added_;
    std::vector<int> byte_ids_;     // id of <0xNN> for each byte, or -1
    std::vector<int> byte_values_;  // byte value of each token, or -1
    std::vector<bool> special_;
    int bos_id_ = -1;
    int eos_id_ = -1;
    int unk_id_ = -1;
    bool add_bos_ = false;
    bool byte_fallback_ = false;
};

#endif // TOKENIZER_H
