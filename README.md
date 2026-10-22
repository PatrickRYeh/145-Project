# IEOR 145/245 Group Simulation: Choosing What to Show 30 Customers

This project answers one question: **when customers arrive one at a time and we only have a few units of each
product, which products should we show each customer to earn the most money?**

We built a model from past customer data, used it to make every decision in the class simulation, and
**earned $270, the maximum possible**.

---

## The problem in plain words

- A store sells **3 products**: product 1 costs **$10**, product 2 costs **$15**, product 3 costs **$20**.
- There are only **6 units of each** (18 units total), and no restocking.
- **30 customers** come in one after another. Each customer is one of three **types** (A, B or C), and we
  learn the type when the customer arrives.
- For each customer we pick which in-stock products to show (this set is called an **assortment**).
  The customer then buys **one** of them or walks away.
- Because there are 30 customers but only 18 units, the goal is really to **sell every unit**, including the
  cheap ones that some customers don't want.

What the past data tell us about each type:

| Type | What they tend to do |
|---|---|
| A | Buys product 3 ($20) or nothing at all |
| B | Strongly prefers product 2 ($15) |
| C | Buys whatever you show them |

So the strategy that works is: **save product 3 for type A, save product 2 for type B, and use type C to sell
whatever else is piling up (especially product 1).**

---

## Where to start

| If you want to... | Open this |
|---|---|
| Read what happened and why it worked | `report/Assortment_Simulation_Report.pdf` |
| See the full analysis, math and experiments | `notebooks/RM_assortment_model.ipynb` |
| Play a game with the model telling you what to show | run `code/play.py` (instructions below) |
| Look up a decision without running any code | `outputs/policy_table.csv` |

---

## Folder layout

```
145 Simulation Project/
│
├── README.md                  ← you are here
├── requirements.txt           ← list of Python packages the code needs
│
├── data/
│   └── large_choice_training_data.csv   ← 63,000 past customers (the input data)
│
├── code/
│   ├── rm_model.py            ← all the math: fit the model, solve for the best policy, test it
│   └── play.py                ← interactive helper used during the live game
│
├── notebooks/
│   └── RM_assortment_model.ipynb        ← step-by-step analysis with explanations and charts
│
├── outputs/                   ← files the notebook creates
│   ├── policy_table.csv       ← the full policy as a lookup table (what to show in every situation)
│   └── figures/               ← charts used in the analysis (PNG images)
│
├── simulation_results/        ← records of the game we actually played
│   ├── LIVE-RHR5BR_assortment_simulation_history.csv   ← official record from the class simulator
│   └── play_py_session_log.txt                         ← what play.py printed during the game
│
├── report/
│   ├── Assortment_Simulation_Report.pdf   ← shareable 5-page report on the game
│   └── simulation_report.html             ← the same report as a web page
│
└── course_materials/          ← provided by the course (not written by us)
    ├── project_instructions.pdf
    ├── lectures/              ← lec09.pdf – lec16.pdf
    └── MNL_code_from_class/   ← the MNL example code from lecture
```

### What each data file looks like

**`data/large_choice_training_data.csv`**: one row per past customer.

| customer_type | assortment | result |
|---|---|---|
| A | {1,2} | 0 |
| C | {2} | 2 |

`assortment` is the set of products that customer was shown. `result` is what they bought
(`0` means they bought nothing).

**`outputs/policy_table.csv`**: one row per possible situation. Find the row matching the customer
number, the current stock of each product, and the arriving customer's type; the `show` column says
what to show.

| customer_number | inv_product1 | inv_product2 | inv_product3 | customer_type | show | expected_revenue_from_here |
|---|---|---|---|---|---|---|
| 1 | 6 | 6 | 6 | A | {1,2,3} | 268.68 |
| 1 | 6 | 6 | 6 | B | {1,2} | 268.68 |

---

## One-time setup

You need **Python 3.10 or newer** (we used 3.13). To check, open a terminal
(on a Mac: press `Cmd + Space`, type **Terminal**, press Enter) and run:

```bash
python3 --version
```

If that prints a version number, you're set. If not, install Python from <https://www.python.org/downloads/>.

Next, go into the project folder and install the packages the code needs:

```bash
cd ~/Desktop/"145 Simulation Project"
python3 -m pip install -r requirements.txt
```

You only have to do this once.

---

## How to play a game with `play.py`

From the project folder, run:

```bash
python3 code/play.py
```

The program then asks you questions, one customer at a time:

```
Customer 1/30 | inventory (6, 6, 6) | revenue so far 0 | expected from here 268.7
  Type (A/B/C, u = undo, q = quit): A
  SHOW {1,2,3}   (predicted: no purchase 0.34, buy 1 0.07, buy 2 0.06, buy 3 0.53)
  Outcome (0 = no purchase, or product number bought): 0
```

1. Type the arriving customer's type (`A`, `B` or `C`) and press Enter.
2. Show the customer exactly the products after `SHOW`. For example, `{1,2,3}` means products 1, 2 and 3.
   The numbers in parentheses are only the model's prediction, not instructions.
3. Type what the customer bought (`0` for nothing, or the product number) and press Enter.

Other keys and options:

- `u` undoes the last customer if you typed something wrong. `q` quits.
- To resume in the middle of a game, for example at customer 12 with stock (3, 4, 2):
  `python3 code/play.py --customer 12 --inventory 3 4 2`
- "revenue so far" + "expected from here" = the model's forecast of your final total.

---

## How to open the analysis notebook

```bash
cd ~/Desktop/"145 Simulation Project"
python3 -m jupyter notebook notebooks/RM_assortment_model.ipynb
```

A browser tab opens with the notebook. It is already run, so you can scroll through the text, tables and
charts without doing anything. To re-run everything yourself, use the menu **Run → Run All Cells**. It
takes about a minute and rewrites the files in `outputs/`.

You can also open the notebook in VS Code, which shows it the same way.

---

## How the model works

1. **Learn how each type behaves.** For each customer type we fit a **multinomial logit (MNL)** model
   (Lecture 14A) to the past data. It gives the probability that a customer buys each product, or nothing,
   for any set of products we show.
2. **Plan ahead with a dynamic program.** The greedy rule from Lecture 14B shows each customer whatever
   earns the most *from that customer*, which leaves the cheap product unsold. Instead, we compute the best
   decision for every possible situation (customer number × stock left × customer type), counting how
   much each unit is worth if we **save it for a later customer** (its *opportunity cost*).
3. **Play the game.** `play.py` looks up the best decision for each customer as the game unfolds.
4. **Check the results.** The notebook compares our policy against simpler rules, and tests it against
   different customer mixes and against errors in the estimated model.

Before the game, the model expected **$268.6** out of the possible $270 (91.5% chance of selling every
unit). Simpler rules do worse: showing everything earns about $267.7, and the greedy rule about $261.0.

---

## About the game we played

The simulator's official record (`simulation_results/LIVE-RHR5BR_assortment_simulation_history.csv`) is
the source of truth.

- Customers **3–27** were shown exactly what `play.py` recommended.
- Customers **1 and 2** were shown `{2,3}` instead of the recommended `{1,2,3}` and `{1,2}`. This cost
  about $0.13 in expected revenue and did not change the outcome.
- All 18 units were sold by customer 27, for **$270**.

The report explains each phase of the game and separates the policy's effect from luck.

---

## Changing the setup

The prices, stock levels and number of customers are set near the top of `code/rm_model.py`:

```python
PRICES = np.array([10.0, 15.0, 20.0])
CAPACITY = (6, 6, 6)
HORIZON = 30
```

Change them, then re-run the notebook and `play.py` to get the new best policy.

---

## Glossary

| Term | Meaning |
|---|---|
| Assortment | The set of products shown to a customer, written like `{1,3}` |
| Customer type | A, B or C; each type has its own buying habits |
| MNL (multinomial logit) | The choice model from class: each product has a "weight", and the chance of buying it is its weight divided by (1 + sum of weights shown) |
| Dynamic program (DP) | A method that solves the game backwards from the last customer, so every decision accounts for the future |
| Opportunity cost | How much expected future revenue a unit is worth if we keep it instead of selling it now |
| Policy | A rule that says what to show in every possible situation |
