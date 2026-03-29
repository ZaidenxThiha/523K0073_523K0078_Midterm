# Part D.2 Required Analysis

## Summary Results Table

| Model | Acc. | Prec. | Recall | F1 | Time/epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP (best configuration) | 0.8800 | 0.8619 | 0.9050 | 0.8829 | 2.50 |
| Vanilla RNN | 0.6677 | 0.6756 | 0.6452 | 0.6601 | 115.84 |
| LSTM | 0.7757 | 0.7691 | 0.7880 | 0.7784 | 135.38 |
| GRU | 0.8268 | 0.7839 | 0.9024 | 0.8390 | 145.57 |

## 1. Which model achieves the highest accuracy? Is the outcome consistent with theoretical expectations?

The highest test accuracy in our experiments is the **best MLP configuration** with **0.8800** accuracy and **0.8829 F1**. The best recurrent result is the **2-layer GRU** with **0.8767** accuracy and **0.8824 F1**, which is very close but still slightly lower in accuracy. Under the fixed one-layer comparison required in Part C, **GRU** is the strongest recurrent variant with **0.8268** accuracy, followed by **LSTM** at **0.7757**, while the vanilla **RNN** is much weaker at **0.6677**.

This is only partly consistent with theoretical expectations. The fixed-configuration recurrent comparison is consistent with theory because **GRU** and **LSTM** are designed to preserve information across long sequences better than a vanilla **RNN**. However, the fact that the best **MLP** slightly outperforms the recurrent models is not impossible. IMDb reviews often contain strong sentiment words, so a mean-pooled embedding model can still perform very well even without modeling word order. The result suggests that lexical sentiment cues are highly informative in this dataset, while the recurrent models may require more tuning or longer training to clearly surpass the MLP.

## 2. Which two models show noticeably different learning-curve behaviour, and what does that suggest?

A clear contrast appears between **Vanilla RNN** and **GRU**. The vanilla **RNN** learns more slowly and reaches a much lower validation and test performance, while the **GRU** converges to a much stronger solution. This suggests that the **GRU** handles long review sequences more effectively and is less affected by the optimization difficulties that hurt the vanilla **RNN**.

Another useful contrast is **GRU (1 layer)** versus **GRU (2 layers)**. The **2-layer GRU** reaches a higher final accuracy (**0.8767** vs **0.8102**) but also takes much longer per epoch (**320.99 s** vs **144.09 s**). This suggests that extra recurrent depth increases representational capacity, but the gain comes with a substantial computational cost.

## 3. Does any model overfit? What evidence supports that conclusion, and what mitigation strategies would you propose?

There is some evidence that the stronger recurrent settings, especially **GRU with 2 layers** and the larger embedding settings, are more prone to overfitting because they have higher capacity and noticeably longer training times. Overfitting would be indicated if training loss keeps decreasing while validation loss stops improving or starts rising, or if training accuracy becomes much higher than validation accuracy.

The dropout experiments in the **MLP** also support this interpretation. Lower dropout can improve fitting, but higher dropout usually reduces overfitting risk by regularizing the model. In our **MLP** results, the differences across dropout settings are modest, which suggests the baseline **MLP** is relatively stable, but recurrent models with larger capacity still need stronger regularization control.

Concrete mitigation strategies:

- increase dropout in the recurrent encoder or classifier
- use early stopping based on validation accuracy or validation loss
- reduce embedding size or hidden size if the model is too large
- reduce the number of recurrent layers when the extra depth does not justify the cost
- train for more epochs only if validation performance is still improving
- tune learning rate and weight decay more carefully

## 4. What common linguistic patterns appear in the misclassified examples?

The misclassified reviews often share these patterns:

- **mixed sentiment**: the review contains both praise and criticism
- **long, discursive writing**: the review spends many words describing plot, history, or background before giving the real opinion
- **positive local phrases inside a negative review**: expressions such as “good”, “beautiful”, “well done”, or “great moment” appear inside an overall negative review
- **contrastive structure**: clauses such as “although...”, “but...”, and “however...” reverse the sentiment later in the sentence
- **subtle or indirect negativity**: the review criticizes realism, pacing, logic, or style without always using very strong negative words

This pattern is visible in several false positives from the best-model error file: some negative reviews contain many positive-looking tokens such as “good”, “beautiful”, “well done”, or “great”, which can bias the classifier toward the positive class even though the overall judgment is negative.

## 5. Why is the MLP blind to word order, and do your results reflect that limitation? Provide a concrete example.

The **MLP** is blind to word order because it uses **mean pooling over token embeddings**. After pooling, the model keeps information about which words are present, but not the sequence in which they appear. In effect, it behaves like a learned bag-of-words model. A sentence and a reordered version of the same words can produce very similar pooled representations.

Yes, the results reflect this limitation. The **MLP** performs very well overall, but many difficult reviews contain **negation**, **contrast**, or **sentiment reversal**, which depend on word order and clause structure. A concrete example is the difference between:

- “the movie is **good**”
- “the movie is **not good**”

The key sentiment change comes from the word **not**, which only works because of its position relative to **good**. Mean pooling weakens that structural relationship. A similar issue appears in long reviews that say positive things about specific aspects before concluding negatively overall. Recurrent models are theoretically better suited to this because they process tokens sequentially and preserve more contextual structure.
