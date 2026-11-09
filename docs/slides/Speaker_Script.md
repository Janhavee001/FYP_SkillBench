# Speaker Script: Skill Library and SkillBench

39 slides. Notes are also in each slide's speaker notes.

---

## Slide 1 · Title

Good morning, Professor. Today I'd like to walk you through where the project stands. It has three parts. First, a library of statistical tools that an LLM agent can call. Second, a set of Claude skills built on top of those tools. And third, a benchmark called SkillBench, which tests whether any of this actually makes the agent more reliable. I'll go through them in that order, and at the end I'll show where the larger second round of the benchmark is right now.

---

## Slide 2 · Overview

Here's the whole project on one slide. The numbers along the top are the four pieces: 61 tools, 8 skills, 206 benchmark questions on 15 datasets, and 137 automated tests.

The idea behind it is simple. When you give an LLM a messy real dataset, it often gives an answer that sounds confident but is wrong, usually because it skipped a step a careful analyst wouldn't skip. So the model gets tested, deterministic tools for the mechanical work, and skills that tell it when and how to use them. Then we measure whether that actually helps.

The table at the bottom is the status. The tools and skills are complete. The skills are tested on two benchmarks: SkillBench, our own benchmark of messy, large and multi-table real data, now in its second, full-scale version; and StatABench, a published benchmark that we run completely unchanged, so the result doesn't rest only on questions we wrote ourselves.

---

## Slide 3 · Part 1: Tool library

So let's start with the foundation, the tool library.

---

## Slide 4 · Tool library at a glance

There are 61 tools across 11 categories, and the chart shows how many are in each. Fifty-two of them come from the original tool catalog, covering the single-table workflow: profiling, cleaning, statistics, testing, modelling, time series and so on. The other nine are a relational layer for working across multiple tables, which I'll come back to.

The key design decision is on the right: there's no language model inside any tool. Every tool is plain, typed Python on top of scipy, statsmodels and scikit-learn. The model decides what to do, and the tool makes sure it's done correctly.

One thing worth pointing out: categories A, grounding and validation, and K, relational analysis, don't exist in any of the five reference systems we looked at. Those are the parts that are new.

---

## Slide 5 · Tools: grounding, loading and descriptive statistics

The next four slides list every tool, so I won't read them all. I'll just point out the shape.

This first one is about understanding the data before touching it. The grounding tools check things like the file's encoding and separator, and whether every column the model refers to actually exists. That's exactly where agents tend to cut corners. Then loading and cleaning, and the basic descriptive statistics.

---

## Slide 6 · Tools: hypothesis testing and modelling

This slide is the core analysis. Hypothesis testing has ten tools, including the t-test with effect size, chi-square, non-parametric tests, bootstrap and power analysis, and multiple-comparison correction. Then regression and modelling: OLS, GLMs, robust and regularised regression, and dimensionality reduction.

---

## Slide 7 · Tools: time series, survival, causal, visualization, interpretation

These are the more specialised methods: time series, survival analysis, causal methods like propensity matching and difference-in-differences, and Bayesian estimates. Then the visualisation tools, and finally interpretation. The interpretation tools turn a result into plain language and check it for over-claiming, for example a significant p-value with a tiny effect size.

---

## Slide 8 · Tools: relational analysis for multiple tables

And this is the relational layer, the nine tools for multi-table data. They find the tables, infer the primary and foreign keys, and plan a join path at the right grain. Then they validate the join: rows before and after, fan-out, orphans. This catches the classic problem where a join silently duplicates rows and inflates your numbers. There's also a check for Simpson's paradox.

---

## Slide 9 · Tool construction

So how is a tool actually built? Every tool follows the same four steps. First it checks its inputs: do the columns exist, are the types right, is there enough data? If not, it returns ok equals false with a reason, rather than crashing or guessing. Second, it computes the statistic with a trusted library. Third, it adds context: effect size, interval, sample size, rows dropped, assumption checks. Fourth, it returns a typed result with the values and any warnings.

The code on the left is a cut-down version of the A/B test tool, so you can see the pattern. The point is that a problem is never silent: if something's off, the agent sees a warning.

---

## Slide 10 · Tool verification

Because the whole point is reliability, the tools themselves have to be trustworthy. There are 137 automated tests, and they all pass. The key rule is that each test compares a tool against an answer the tool didn't produce itself.

The table shows the kinds of evidence. Some tests use known benchmark values, like the Titanic survival rate. Some check agreement with scipy or scikit-learn to nine decimal places. Some use published textbook results, like the classic Freireich leukaemia survival data, matched against R. Others plant a known structure in synthetic data and check that it's found, or reproduce real failures we saw earlier, like a unit slip of 38.38 versus 0.3838. The chart on the right shows how the tests are spread across the test files.

---

## Slide 11 · Part 2: Skills

That's the toolbox. But a toolbox alone isn't enough, because the model also has to know when to use what. That's what the skills are for.

---

## Slide 12 · The eight skills

There are eight skills, and each one covers a type of analysis: data profiling, distributions, correlation, group comparison, regression, time series, visualisation, and relational multi-table analysis. Underneath each one you can see how many tools it documents.

A good way to think of a skill is as the playbook an experienced analyst would follow for that kind of question: what to check first, which method fits, and what mistakes to avoid. Each one is a single package you can upload directly to the Claude app.

---

## Slide 13 · Skill structure

Each skill follows Anthropic's Agent Skills format. The table shows the sections of the SKILL.md file. The name and description at the top are what the model matches against the question to decide whether to use the skill. Then come when to use it and when not to, setup, a numbered procedure where each step names the tool to call, the common errors, how to report the result, and a worked example.

On the right is what's inside the zip: the SKILL.md, a generated tool reference, and the tool library itself, bundled in. So the skill is completely self-contained. And at the bottom right is what happens in the Claude app: the description matches, Claude reads the skill, runs the tools in its sandbox, and reports in the skill's format.

---

## Slide 14 · Skill-tool relationship

This table shows how the two layers connect. Each row is a skill, each column is a tool category, and the numbers are how many tools from that category the skill documents. You can see, for example, that almost every skill draws on grounding and descriptive tools, and that the relational skill is the main user of category K.

One important detail at the bottom: every skill bundles the full library. So even if a question needs a tool the skill doesn't mention, it's still available. The tools are the how, and the skills are the when and why.

---

## Slide 15 · Skill construction and validation

This is how the skills are built. The SKILL.md is written by hand. The tool reference is generated from the code's own signatures and docstrings, so it can never drift out of date. Then the library is copied in, every example is run from the zip in a clean process, and it's packaged. One build command produces all eight.

On the right is a real test in the Claude app. I asked, "Did survivors pay more?" without naming any skill, and it loaded group-comparison by itself. It reported that survivors paid about 26 pounds more, with a confidence interval of roughly 19 to 34 pounds and a medium effect size. And importantly, it reported this as an association, noting that fare tracks passenger class, not as a causal claim.

---

## Slide 16 · Part 3: SkillBench

So now the important question: does any of this actually help? That's what the benchmark is for.

---

## Slide 17 · Motivation

The reason we needed a new benchmark is the number on the left. On the phase-one questions, a plain LLM already got method selection right 98 percent of the time. So that benchmark couldn't tell the systems apart.

The data there, Titanic and Wine, is small, clean, and very well known. But in practice the errors that really matter aren't about choosing the method. They're about handling the data: the wrong grain, missing-value codes, joins that silently duplicate rows, confounding.

So SkillBench targets exactly those. The table lists the design requirements: real data with up to three million rows, uncleaned; questions where a trap changes the answer; reference answers computed independently of the tools; the same model and wording in every condition; and proper statistics.

---

## Slide 18 · Benchmark versions

The benchmark has been run in two versions, and both are kept here so the progression is clear.

Version one was the pilot: 108 questions on nine datasets, one model, three conditions, and one run each, so 324 answers in total. It showed a clear effect, but a pilot like that has limits. One run doesn't tell you how much results vary, and one model doesn't tell you whether the effect generalises.

Version two is the full-scale version. The question bank nearly doubles to 206 on 15 datasets. There's a fourth condition, the oracle skill, and a second, smaller model. Each agent condition is run three times at temperature 0.7, and an independent judge from a different model family does the scoring. That comes to 4,120 answers, more than twelve times the pilot, as the chart shows.

---

## Slide 19 · Benchmark setup: datasets

These are the eleven real datasets, and they carry the headline results. They're public, some of them millions of rows, and kept exactly as published, mess included. The last column shows what makes each one hard. NYC taxi data has trips dated outside the month and cash tips that are never recorded. The flights data only records delay causes for flights more than 15 minutes late, and one of its lookup files is in Latin-1 encoding. Olist has a hidden customer key, and F1DB has a 47-table schema.

On top of these there are four small synthetic databases, used only to test specific situations in a controlled way. They're reported separately as an appendix.

---

## Slide 20 · Benchmark setup: question bank

Across those datasets there are 206 questions: 183 on real data and 23 synthetic. The chart on the left shows how many questions each dataset contributes. On the right, the top chart is the difficulty split, 65 easy, 86 medium and 55 hard, and the bottom chart shows the question categories, so you can see the mix of analysis types.

---

## Slide 21 · Question construction

Every question goes through the same five-step pipeline. First, we profile the real data to find its genuine problems. Second, we write a question that has one unambiguous answer. Third, we compute the reference answer in plain pandas and scipy, never with our own tool library, because otherwise we'd be grading the tools against themselves. Fourth, we also compute the careless answer, the one you get by taking the shortcut, and if the trap doesn't actually change the answer, we drop it. Fifth, we audit every reference and reword anything ambiguous.

The table shows a real example, FL05 from the flights data. It asks for the average weather delay per flight. The correct answer is 0.99 minutes, because a blank means zero delay. If you just average the recorded values, you get 4.89 minutes, nearly five times too high. A gap that size tells you exactly which path the model took.

---

## Slide 22 · Trap taxonomy

140 of the 206 questions contain a trap, and they fall into five families, shown in the chart: grain and unit of analysis; joins and linkage; missing and unrecorded values; dirty or invalid records; and statistical reasoning.

The table gives concrete examples. Computing median invoice value from invoice lines gives you 10 pounds instead of 304. Using the wrong customer key in Olist makes repeat customers look like zero percent instead of 3.1. And forgetting to convert currency units turns 0.85 million dollars of revenue into 9.1 million. Each of these is something a careful analyst checks automatically, and an LLM tends to skip.

---

## Slide 23 · Experimental design

The four conditions each add one thing. Plain writes one script, runs it once, and answers from its output. That's the weak baseline. Agent can run Python up to ten times before it submits, which is the strong baseline. Agent plus skills has the same loop, plus the ability to load a skill and the bundled tool library, and it has to choose the skill itself. And oracle is the same, but with the correct skill already loaded.

The table on the right is the v2 setup: two models, gpt-oss-120b and the smaller 20b, three runs per agent condition, temperature 0.7 with fixed seeds, and qwen as the judge. That gives 4,120 answers on SkillBench, and the same design on StatABench gives another 3,960.

The key comparison is oracle against agent plus skills. The difference tells us how much we lose purely because the model picks the wrong skill or none at all. And comparing the two models tells us whether skills help a weaker model more.

---

## Slide 24 · Scoring and statistics

On scoring: the primary measure is correctness, and it's fully deterministic. A number has to be within tolerance, and a label has to match. We also record a trap fall, meaning the wrong answer is exactly the careless answer we precomputed. The judge model only scores analytic quality, things like whether the method fits and whether the caveats are there. We also track skill selection and cost.

For the statistics, accuracy is the mean of the runs, uncertainty is a bootstrap 95 percent confidence interval over questions, and conditions are compared with a paired permutation test.

At the bottom: before reporting v1, we audited it. Category matching was relaxed for equivalent names, a few answers were recovered from transcripts, and three ambiguous questions were reworded and rerun. Every fix was applied equally to all conditions.

---

## Slide 25 · Preliminary results (v1)

These are the v1 results: 108 questions, gpt-oss-120b, one run per condition. On the left, overall accuracy goes from 86 percent for plain, to 93.5 for agent, to 95.4 with skills.

On the right is the more interesting picture, by difficulty. On easy questions every condition does well. On hard questions it separates: 68 percent for plain, 80 for agent, and 92 with skills. So the skills earn their place exactly where the analysis gets tricky. The confidence intervals are at the bottom.

---

## Slide 26 · Preliminary results (v1): comparisons

These are the paired comparisons, using McNemar's test. Plain against agent with skills: 11 questions only skills got right, against 1 the other way, p equals 0.006, so that's significant. Plain against agent is also significant, at 0.039. But agent against agent with skills is only 4 versus 2, p of 0.69. So over a strong agent, the gain is small and not significant in v1.

The table on the right explains a big part of why. With skills available, a skill was loaded for only 52 percent of questions, and the fitting skill for only 39 percent. So half the time, the skills weren't even used. Skill selection is the bottleneck, and that's exactly what the oracle condition in v2 measures. And honestly, one run on 108 questions is too small to detect a two-point difference, which is the main reason for v2.

---

## Slide 27 · SkillBench v2 run

This slide is the v2 run itself. All 4,120 answers are complete. The chart shows that every model and condition got its full set: one run for plain, and three runs each for agent, agent with skills, and oracle, on both models.

One thing I need to be upfront about. While auditing the transcripts, I found a bug in our harness. gpt-oss sometimes leaks its internal chat-format tokens into tool calls, so a call would arrive named something like run_python followed by a channel token, or with the code not wrapped in JSON. Our first harness parsed those strictly, so it ran an empty script and told the model its code printed nothing. For the small model that hit up to half its code calls in some runs.

So I fixed the parser, applied the same fix to every condition, and moved every affected answer to a quarantine folder and reran it, about 4,600 answers across both benchmarks. A final scan of every transcript finds no affected answers left. So the numbers on the next slides all come from the fixed harness.

---

## Slide 28 · Results (v2): accuracy

These are the final v2 results on the 181 real-data questions, averaged over three runs, with 95 percent confidence intervals.

For the large model, gpt-oss-120b, all four conditions sit close together: 90.6% plain, 93.0% agent, 92.8% with skills, and 91.9% with the oracle skill. It's near the ceiling on this benchmark, so there's not much room for skills to help, and the intervals all overlap.

The small model, gpt-oss-20b, is where it gets interesting: 77.3% plain, 75.5% agent, 80.5% with skills, and 78.8% with the oracle. So for the weaker model, skills lift it by about five points over the plain agent. Notice also that for the small model, the agent loop on its own doesn't beat one-shot plain analysis. It's the skills condition that makes the difference.

---

## Slide 29 · Results (v2): paired comparisons

This table puts numbers on those differences. Each comparison is paired, so both conditions are judged on exactly the same questions, and the p-value comes from a permutation test.

For the large model, nothing is significant. Adding skills to the agent changes accuracy by -0.2 points, p equal to 1.000. That's essentially zero.

For the small model, adding skills gives +5.0 points, with p equal to 0.012. That's the significant skills effect on SkillBench.

The oracle row is about selection. Giving the model the right skill instead of letting it choose changes accuracy by -0.9 points for the large model and -1.7 for the small one, and neither is significant. So being handed the right skill doesn't help more than choosing. I'll come back to why that's surprising in a moment.

---

## Slide 30 · Results (v2): difficulty

Broken down by difficulty, the differences sit in the hard questions.

For the small model on hard questions, accuracy is 63% plain, 62% agent, 70% with skills, and 67% with the oracle. So skills help most where the analysis is tricky, the same pattern we saw in v1.

For the large model it goes the other way on hard questions: 87% plain and 89% agent, but 83% with skills and 82% with the oracle. There are only 46 hard questions, so this isn't significant, but it suggests that for a strong model a generic procedure can sometimes pull it away from an analysis it would have got right.

---

## Slide 31 · Results (v2): traps and multi-table questions

On the left are questions with and without a trap, and on the right, single-table against multi-table questions.

For the small model, both traps and multiple tables cost accuracy, and skills recover part of that. On the 100 multi-table questions it goes 74% → 78% → 86%, from agent, to skills, to skills with the validator. For the large model, multi-table questions sit at 90%–93% in every agent condition, so it already handles the joins well.

Trap falls, meaning a wrong answer that's exactly the careless answer we precomputed, are low overall: 4% for the large model's agent and 4% for the small model's. The traps that do catch models are a few specific ones, which I'll come back to on the inference slide.

---

## Slide 32 · Results (v2): skill selection, quality and cost

This slide has three parts.

On the left is skill selection. The large model loaded a skill for 38% of questions, and the fitting one for 30%. The small model loaded one for only 9%. And accuracy is the same whether or not a skill was loaded: 80% against 80% for the small model. So the small model's gain can't come from reading the skill text, because it almost never reads it. It must come from something else in the skills condition: the list of skills in the prompt, which reminds it what to check, or the tested library being available. Separating those is the next ablation.

In the middle is the judge's quality rating. Method fit is high everywhere, but caveats drop sharply once the model works as an agent: from 87% in plain to 39% with skills for the large model. Agents report a number and stop. That's something the skills' report format should fix.

At the bottom is cost. Skills roughly double prompt tokens, from 9.1 thousand to 19.0 thousand per question for the large model, because the skill catalog and skill text go into the conversation.

---

## Slide 33 · External benchmark: StatABench

SkillBench is our own benchmark, so the obvious question is whether the effect holds on someone else's. For that we use StatABench, a published 2026 benchmark for statistical analysis, and one of the two benchmarks we ran in phase one.

We take its 198 Stat-Closed items that need data and tools, completely unchanged: the same questions, the same datasets, and StatABench's own reference answers. The chart shows how they spread across 17 statistical tasks, from correlation and contingency tables to GLMs and survival analysis.

Grading also uses StatABench's own comparison prompt, copied word for word, so we don't grade with rules we wrote ourselves. The only change is the judge model, qwen, which comes from a different family from the models under test.

For context, the last row shows that StatABench's own agent, with its 36 statistical tools, scored 66 percent with gpt-oss-120b in phase one. That leaves real headroom, unlike SkillBench, where the big model is near the ceiling. And it's worth noting that even this published benchmark generates many of its items with an LLM: 87 of the 198 came from Gemini.

---

## Slide 34 · Results: StatABench

Now the external benchmark, StatABench, on its own questions with its own grading, averaged over 3 runs.

First, a sanity check: the large model's agent scores 64.5%, close to the 66 percent that StatABench's own 36-tool agent got in phase one. So our harness is in the right range.

For the large model, all conditions are again close: 61.6% plain, 64.5% agent, 63.8% with skills, and 64.8% with the oracle. No difference is significant. So even with real headroom, about a third of answers wrong, skills don't move the large model.

For the small model, the order is the same as on SkillBench: 48.5% plain, 54.4% agent, 56.6% with skills, and 59.1% with the oracle. Skills over the agent is +2.2 points, p equal to 0.383, so not significant on its own, but skills over plain is +8.1 pts, p = 0.011. So the direction replicates on a benchmark we didn't write, but the effect is smaller.

---

## Slide 35 · Results: StatABench by task

By task, the picture is noisy, because many tasks have fewer than ten items. For the large model, the biggest gains from skills are in high-dimensional (+27.8 pts), non-parametric (+16.7 pts), survival analysis (+10.3 pts). For the small model, they're in confidence interval (+29.6 pts), exploratory analysis (+16.7 pts), contingency table test (+15.0 pts). But those are small groups, so I wouldn't read much into any single bar.

What's more useful is where every condition fails. Variance tests stay below a third in every condition, and GLMs and survival analysis stay around 30 to 55 percent, which matches the weak spots of StatABench's own agent in phase one. Those are the clearest targets for better skills.

On selection, the large model loaded a skill for 67% of items here, much more often than on SkillBench, and a fitting one for 47%.

---

## Slide 36 · Validation loop (RQ3)

This slide answers RQ3: does a validation step catch and repair errors?

The validator starts from exactly the same first answers as the skills condition, so the comparison is fully paired. It runs two deterministic checks, did the code actually run, and does the submitted number appear in the code's output, plus a checklist critic covering grain, population, missing codes, joins, method and measurability. If anything is flagged, the agent gets up to four more code runs to fix it.

For the small model it clearly helps, on both benchmarks. On SkillBench it flags 65% of the wrong answers, fixes 39 and breaks only 5, a gain of +6.3 points, which is significant. On StatABench the gain is +3.9 points, also significant.

For the large model it's neutral. On SkillBench it fixes 15 and breaks 13, and on StatABench it fixes 23 and breaks 27, with a false-alarm rate of 30% on correct answers. So when the model is already strong, the critic's doubts are as likely to talk it out of a right answer as into one.

One caveat: the critic is the same model checking its own work. A critic from a different model family might do better, and that's a clear next experiment.

---

## Slide 37 · Reference-answer validation

Before the results can be trusted, the answer key has to be right, so this slide covers how we checked it. There are four independent checks.

First, every reference answer is computed with plain pandas, scipy and statsmodels, and never with the tool library being tested. So the key can't favour the skills condition.

Second, we checked the data itself against the publishers. For all 11 record counts the publishers state, the benchmark's copy matches exactly. That rules out broken downloads or parsing.

Third, a consensus audit on the final data. Each question has 24 agent answers across models, runs and conditions. If at least half agree on an answer that isn't the reference, the question is flagged and checked by hand. 13 of 206 were flagged. In 10, the reference holds and the consensus fell into the trap, mostly giving exactly the careless answer. In 2, the consensus used a wrong method. And 1 has two defensible readings, so it's excluded from all conditions, along with one more found earlier. Three others were correct answers given at lower precision, and a rounding rule now accepts those in every condition.

Fourth, I reviewed a random sample of 30 questions myself, reading the exact code behind each reference. 29 were clearly right, and the one I was unsure about turned out to be right when I checked how the data was built. So 30 out of 30. A second, independent reviewer is still needed.

And one more safeguard: the models being tested, the model that helped write the answer key, and the judge all come from three different model families.

---

## Slide 38 · Inference

So what can we conclude?

RQ1, do skills improve correctness? For the small model, yes. On SkillBench it's +5.0 pts, p = 0.012 over the agent. On StatABench the gain over the agent is +2.2 pts, p = 0.383, not significant on its own, but +8.1 pts, p = 0.011 over plain. For the large model there's no change on either benchmark.

RQ2, does the agent select skills? Rarely. The large model loads one for 38% of SkillBench questions and 67% on StatABench, and the small model for only 9% and 27%. And giving it the right skill doesn't add a significant gain. So the effect we see for the small model isn't coming from reading skill text, and explaining it is the open question for the next phase.

RQ3, does validation catch errors? For the small model, yes, on both benchmarks: +6.3 pts, p < 0.001 on SkillBench and +3.9 pts, p = 0.013 on StatABench. For the large model it's neutral.

And on multi-table questions, the small model goes 74% → 78% → 86% from agent to skills to the validator, while the large model already handles the joins.

The overall message is at the bottom. Skills and validation help where the model is weak, not where it's already near its ceiling. The traps that still catch most runs are unrecorded values, Simpson's paradox and censoring. And agents tend to drop the caveats a careful analyst would state.

---

## Slide 39 · Status and next steps

To finish, the status. SkillBench v2, StatABench and the validation loop are all complete, with full statistics. The answer key has passed four checks, and the remaining item there is a second reviewer.

On the right are the next steps. The most important is an ablation to explain why the small model gains from skills it rarely loads: is it the list of skills in the prompt, the tested library, or the skill text? After that, a validator critic from a different model family, new skills for unrecorded values, censoring and Simpson's paradox, and requiring caveats in the skills' report format. And for robustness, pinning the inference provider, and adding DataSciBench and a second model family.

That's everything. Thank you, and I'm happy to take any questions.
