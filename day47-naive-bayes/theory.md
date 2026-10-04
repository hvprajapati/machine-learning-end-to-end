# Naive Bayes: Theory

> Read this first, then open `01-naive-bayes-classification.ipynb`.

---

## 1. The idea

Naive Bayes classifies using **probability**. For a new example it asks, for every class:

> *"How likely is this class, given what I see?"*

and picks the class with the **highest probability**.

It is called:
- **Bayes**, because it uses **Bayes' theorem**;
- **Naive**, because it makes one simplifying assumption: features are **independent** of each other within a class.

---

## 2. Probability you need (only this)

| Notation | Read as | Example |
|---|---|---|
| $P(A)$ | probability of A | $P(\text{spam}) = 0.4$: 40% of emails are spam |
| $P(A \mid B)$ | probability of A **given** B is true | $P(\text{"free"} \mid \text{spam}) = 0.75$: 75% of spam emails contain "free" |

Note that $P(A \mid B)$ and $P(B \mid A)$ are **different**: "most spam contains *free*" is not the same as
"most emails containing *free* are spam".

---

## 3. Bayes' theorem

$$
P(\text{class} \mid \text{data}) = \frac{P(\text{data} \mid \text{class}) \times P(\text{class})}{P(\text{data})}
$$

| Part | Name | Meaning |
|---|---|---|
| $P(\text{class} \mid \text{data})$ | **posterior** | what we want: the probability of the class after seeing the data |
| $P(\text{data} \mid \text{class})$ | **likelihood** | how typical this data is for that class |
| $P(\text{class})$ | **prior** | how common the class is overall |
| $P(\text{data})$ | **evidence** | the same for every class, so we can ignore it when comparing classes |

So for classification we only need:

$$
P(\text{class} \mid \text{data}) \;\propto\; P(\text{data} \mid \text{class}) \times P(\text{class})
$$

($\propto$ means "is proportional to". Compute this score for each class, then divide by the total so they add up to 1.)

---

## 4. The "naive" assumption

With several features, $P(\text{data} \mid \text{class})$ becomes $P(x_1, x_2, \dots \mid \text{class})$, which is hard
to estimate. Naive Bayes assumes the features are **independent within each class**, so we can simply **multiply**:

$$
P(\text{class} \mid x_1, x_2, \dots, x_n) \;\propto\; P(\text{class}) \times P(x_1 \mid \text{class}) \times P(x_2 \mid \text{class}) \times \dots \times P(x_n \mid \text{class})
$$

This is rarely exactly true (age and salary are related, for example), but the classifier still works surprisingly
well in practice, because we only need the **ranking** of the classes to be right, not the exact probabilities.

---

## 5. Worked example: a spam filter by hand

We have **10 emails**: 4 spam, 6 not spam ("ham").

| | Spam (4) | Ham (6) |
|---|:-:|:-:|
| contains "free" | 3 | 1 |
| contains "meeting" | 1 | 4 |

**Priors:** $P(\text{spam}) = 4/10 = 0.4$, $P(\text{ham}) = 6/10 = 0.6$

**Likelihoods:**
- $P(\text{free} \mid \text{spam}) = 3/4 = 0.75$, $P(\text{free} \mid \text{ham}) = 1/6 \approx 0.167$
- $P(\text{meeting} \mid \text{spam}) = 1/4 = 0.25$, $P(\text{meeting} \mid \text{ham}) = 4/6 \approx 0.667$

### Example A: an email containing "free"

| Class | Score = prior × likelihood |
|---|---|
| spam | $0.4 \times 0.75 = 0.30$ |
| ham | $0.6 \times 0.167 = 0.10$ |

Normalise: $P(\text{spam} \mid \text{free}) = \frac{0.30}{0.30 + 0.10} = \mathbf{0.75}$ → **spam**.

### Example B: an email containing both "free" and "meeting"

Multiply the likelihoods (the naive assumption):

| Class | Score |
|---|---|
| spam | $0.4 \times 0.75 \times 0.25 = 0.075$ |
| ham | $0.6 \times 0.167 \times 0.667 = 0.067$ |

$P(\text{spam}) = \frac{0.075}{0.075 + 0.067} \approx \mathbf{0.53}$ → **spam, but only just**. The word "meeting"
pulled it strongly toward ham.

*(For simplicity we used only the words that appear in the email.)*

---

## 6. Types of Naive Bayes

The only difference between the types is **how they compute $P(x \mid \text{class})$**:

| Type | Features | How $P(x \mid \text{class})$ is computed | Typical use |
|---|---|---|---|
| **GaussianNB** | numeric (age, salary, …) | height of a **bell curve** (normal distribution) with that class's mean and variance | numeric data, as in our notebook |
| **MultinomialNB** | counts (how often each word appears) | word frequencies in that class | text classification, spam filters |
| **BernoulliNB** | yes/no (word present or not) | share of class examples with that feature | short texts, binary features |

### Gaussian Naive Bayes (used in the notebook)
For each class and each feature, training stores a **mean** $\mu$ and **variance** $\sigma^2$. The likelihood of a
value $x$ is the height of the bell curve at $x$:

$$
P(x \mid \text{class}) = \frac{1}{\sqrt{2\pi\sigma^2}}\; e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

What the model learned on the Social Network Ads data:

| Class | Prior | Mean age | Mean salary |
|---|:-:|:-:|:-:|
| did not buy | 0.63 | 33.5 | 60,180 |
| bought | 0.37 | 46.0 | 85,595 |

For a **30-year-old earning 87,000**: the age is very typical of non-buyers, which outweighs the slightly buyer-like
salary, so $P(\text{did not buy}) \approx 0.90$. The notebook computes this by hand and gets the same number as
scikit-learn.

---

## 7. The zero-probability problem and smoothing

If a word never appeared in spam during training, $P(\text{word} \mid \text{spam}) = 0$, and multiplying by 0 makes
the whole spam score 0, no matter what other words say.

**Fix: Laplace (add-one) smoothing:** add 1 to every count, so nothing is ever exactly 0.
`MultinomialNB(alpha=1.0)` does this by default (`alpha` is the amount added).

---

## 8. Does Naive Bayes need feature scaling?

**No.** Each feature is handled **on its own** by its own distribution. Naive Bayes never combines features into a
distance (unlike K-NN or SVM). In the notebook, the predictions with and without scaling are **identical**.

---

## 9. Using Naive Bayes in scikit-learn

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)

model.predict(X_test)          # class labels
model.predict_proba(X_test)    # probability of each class
model.class_prior_             # priors
model.theta_, model.var_       # mean and variance of each feature per class
```

```python
# Text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

vec = CountVectorizer()
X_counts = vec.fit_transform(messages)
model = MultinomialNB().fit(X_counts, labels)
model.predict(vec.transform(["win a free prize"]))
```

---

## 10. Strengths and limitations

| Strengths | Limitations |
|---|---|
| **Very fast** to train and predict (just counting and averages) | The independence assumption is usually not true |
| Works well with little data | Probability values can be over-confident (too close to 0 or 1) |
| Excellent for **text** (spam, sentiment, topic) | GaussianNB assumes bell-shaped features |
| Handles many features and multiple classes easily | Usually less accurate than tuned SVM / ensembles on numeric data |
| No scaling needed, almost nothing to tune | Correlated features get "double-counted" |

---

## 11. Naive Bayes vs the other classifiers

| | **Naive Bayes** | **Logistic Regression** | **K-NN** | **SVM** |
|---|---|---|---|---|
| Approach | probability (Bayes) | weighted sum + sigmoid | neighbours vote | widest margin |
| Training speed | fastest | fast | none (stores data) | slow on big data |
| Needs scaling | no | recommended | yes | yes |
| Gives probabilities | yes | yes | yes (vote share) | not by default |
| Test accuracy here | 0.90 | 0.89 | 0.93 | 0.90 (rbf: 0.93) |

---

## 12. Quick revision questions

1. *What does Naive Bayes compute for each class?* $P(\text{class}) \times$ the product of $P(\text{feature} \mid \text{class})$.
2. *Why "naive"?* It assumes features are independent within each class.
3. *What is a prior?* How common a class is before looking at the features.
4. *Which type for numeric features? For word counts?* GaussianNB; MultinomialNB.
5. *What is Laplace smoothing for?* To avoid zero probabilities for words never seen in a class.
6. *Does it need scaling?* No; each feature is modelled separately.

---

## Summary

- Naive Bayes picks the class with the highest **posterior probability**, using **Bayes' theorem**.
- Score per class = **prior × likelihood of each feature**, multiplied together (the naive assumption).
- **Gaussian** NB for numbers, **Multinomial** / **Bernoulli** NB for text.
- Very fast, needs no scaling, and is a strong baseline, especially for text.
