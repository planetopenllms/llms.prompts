# Q: does byte pair encoding work with unicode characters or with true (8-bit) bytes?

A: **Byte Pair Encoding (BPE)**, as originally defined, operates on **bytes**, not directly on Unicode characters. Specifically, BPE works with **raw byte sequences** (i.e., sequences of 8-bit characters). However, when applied in modern NLP contexts, BPE is adapted to work with **Unicode characters** by treating each Unicode character as a distinct unit, and performing the encoding at the **character or subword level**, rather than at the byte level.

Here’s a more detailed breakdown of the differences and how BPE adapts to modern use:

### Original BPE (Byte-Level)
- **Byte Pair Encoding** was originally developed for **data compression**, where it worked directly on **bytes**, which are 8-bit units. This approach repeatedly merged the most frequent pair of adjacent bytes in a sequence until a predefined vocabulary size was reached.
- BPE, in its original form, was not specifically designed to handle text processing or Unicode, and instead, it operated on raw binary byte sequences. It could be used on any kind of data, not just text.

### BPE in NLP (Modern Adaptation)
In modern NLP, BPE has been adapted to work with **textual data**, which is often encoded using Unicode (especially UTF-8, UTF-16, etc.). Here's how BPE is typically applied in this context:

1. **Unicode Characters as Units**: In text processing, each **Unicode character** is often treated as a "token" for the purpose of BPE. Unicode characters can be encoded in UTF-8 or other encodings, where each character may span one or more bytes. In practice, though, BPE typically operates at the **character level**, treating each Unicode character as a distinct unit, even if it takes multiple bytes in UTF-8 encoding.

2. **Handling Subwords**: In NLP, BPE is typically used to split text into subword units, and it often works on the **characters** of the word. It will iteratively merge the most frequent adjacent pairs of characters (or subwords) based on their frequency in a given training corpus, rather than treating them as raw 8-bit byte sequences. This way, it can handle out-of-vocabulary (OOV) words by splitting them into smaller subword units.

3. **Unicode and Encoding Considerations**:
   - When applying BPE to text, the first step is to convert the text into a sequence of **Unicode code points** or a specific **encoding like UTF-8**.
   - The BPE algorithm will then tokenize these code points or characters. For example, when BPE merges characters like "un" and "happiness", it might create a new token, such as "unhappi", using subword units instead of raw byte pairs.

4. **BPE vs Byte-level Tokenization**:
   - Modern implementations of BPE in NLP, such as those in models like **GPT-2** and **BERT**, often use tokenizers like **Byte-level BPE** (e.g., in **GPT-2**), where the input text is first encoded into bytes (typically in **UTF-8** encoding), and then BPE operates on these byte sequences.
   - This method allows the model to handle **character-based tokenization**, and also deals effectively with multilingual data and special characters (such as punctuation, emojis, or non-Latin scripts).

### Example of BPE with Unicode:
- When using a **Unicode**-based tokenization scheme like BPE, each character (or subword unit) can be represented using a code point (like `U+0041` for the letter "A").
- BPE then operates on these code points (characters) to merge frequently occurring pairs into new subword units. This enables the model to handle out-of-vocabulary words more effectively by breaking them down into familiar subword units.

### Conclusion
- **Original BPE** works directly with raw **bytes** (8-bit sequences).
- **Modern BPE** in NLP typically operates on **Unicode characters or subword units** that are encoded into bytes (using UTF-8 or similar encodings), but the merging process happens at the character or subword level, not at the raw byte level.
- The modern usage of BPE in NLP (like in **GPT-2**) often works with **byte-level tokenization**, where text is first converted into bytes (UTF-8 encoded), and then BPE merges these byte sequences. However, the underlying goal is to handle subword units, not just individual 8-bit bytes.