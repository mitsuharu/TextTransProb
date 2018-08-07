# TextTransProb

- 文字列中の文字頻出パターンを学習して，文字列の生成確率を計算する
- 任意の文字列が学習データに類似しているかどうかを評価する

## 使い方

学習

```
training_file = "./samples/en_words.txt"
model_file = "./samples/model.pki"

tp = TransProb()
tp.train(training_file=training_file, save_file=model_file)

print("p =", tp.calc_prob("pen"))
```

学習済みデータの読み込み

```
tp = TransProb(model_file="./samples/model.pki")
print("p =", tp.calc_prob("pen"))
```

---

# TextTransProb

- It trains frequent patterns of characters in character strings
- It computes a transition probability of a character string.
- It evaluates whether arbitrary character string is similar to training data.
