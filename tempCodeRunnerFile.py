tokenizer = Tokenizer()
tokenizer.learn(vocab, merge_rules, data.head(1))