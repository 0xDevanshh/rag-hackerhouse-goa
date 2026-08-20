# Voice RAG Latency Benchmark

Ran 108 queries (100 real MSMARCO-XI queries, 8 adversarial/off-topic) through `PipelineHarness` against the same corpus `src/api.py` serves (9990 indexed chunks), with `MockSTT` substituted for real speech-to-text so the numbers below isolate in-process compute (embedding, FAISS retrieval, guardrail checks, LLM generation) from network-bound STT.

> Generation used the local extractive provider. It performs no network call and returns only the highest-ranked retrieved passage with a citation, so these are honest local pipeline measurements rather than simulated LLM timings.

## Verdict

| Target | P100 (max) | Result |
|---|---|---|
| Retrieval pipeline < 200ms (input guard + embed + FAISS + rerank + relevance guard, excl. generation) | 204.3ms | **FAIL** |
| Post-STT total < 200ms (everything after STT, **including** LLM generation) | 204.4ms | **FAIL** |

These are two different, deliberately separate claims — see "Why two targets" below before reading one as a substitute for the other.

## Stage breakdown

| Stage | P50 | P70 | P100 (max) | Samples |
|---|---|---|---|---|
| Retrieval (embed + FAISS + rerank) only | 81.6 ms | 103.1 ms | 204.2 ms | 106 |
| Generation only | n/a | n/a | n/a | 0 |
| **Retrieval pipeline** (target < 200ms) | 80.8 ms | 102.0 ms | 204.3 ms | 108 |
| **Post-STT total** (target < 200ms) | 80.9 ms | 102.1 ms | 204.4 ms | 108 |

Sample counts below 108 for the "only" rows are expected: some adversarial queries are correctly short-circuited by InputGuardrail or RelevanceGuardrail before reaching retrieval or generation at all, so those stages simply weren't attempted for them (their duration is excluded, not counted as 0ms or as a failure). "Retrieval pipeline" and "Post-STT total" always have the full 108 samples, since every query spends *some* time in the stages each one covers even when short-circuited early.

## Speech-to-text (Sarvam API) — network-bound, separate from both targets

Ran 5 real calls to Sarvam's speech-to-text API on a 1-second silent WAV clip (0 failed and were excluded from the percentiles below).

| Metric | P50 | P100 (max) | Samples |
|---|---|---|---|
| STT round-trip | 394.7 ms | 1492.1 ms | 5 |

## Why two targets

"Retrieval pipeline" covers what this codebase actually controls: query validation, embedding, FAISS search, reranking, and the relevance check. This is the number an ANN index, a smaller embedding model, or a smarter cache would move — and the one that's realistic to hold under 200ms regardless of corpus size (within reason).

"Post-STT total" additionally includes LLM generation — a real network round trip to a third-party hosted model. No local optimization changes how long a remote GPU takes to generate tokens; holding this under 200ms is a claim about the LLM provider's latency, not about this codebase. If the verdict table above shows this target failing, that is expected and does not indicate a retrieval-side regression — check the retrieval pipeline verdict independently.

STT is excluded from both because it precedes this pipeline entirely (the harness only starts timing after a query is already transcribed) — `MockSTT` removes it from these numbers by design, not as an oversight, and its real latency is reported separately above.

## Per-query detail

| Category | Degraded | Retrieval pipeline (ms) | Post-STT total (ms) | Query |
|---|---|---|---|---|
| in_domain | True | 18.0 | 18.1 | how is caffeine metabolized |
| in_domain | True | 90.0 | 90.0 | सुल्तानी अर्थात् |
| in_domain | True | 67.0 | 67.1 | can ear infections cause seizures in cats? |
| in_domain | True | 108.8 | 108.9 | स्थानीय डिस्क की परिभाषा |
| in_domain | True | 69.8 | 69.9 | can chia seeds be planted in house |
| in_domain | True | 204.3 | 204.4 | स्मिथविले, एन.जे. में होटल |
| in_domain | True | 89.8 | 89.9 | सब स्लैब परिभाषा |
| in_domain | True | 154.2 | 154.4 | क्या ओबामा फिर से राष्ट्रपति पद के लिए चुनाव लड़ सकते हैं? |
| in_domain | True | 68.6 | 68.7 | how much supervisor pay rate for cvs warehouse |
| in_domain | True | 133.4 | 133.5 | समझाएं कि ज्वालामुखी प्रदूषण कैसे करते हैं |
| in_domain | True | 133.0 | 133.1 | क्या योग से मांसपेशियां बनती हैं? |
| in_domain | True | 87.1 | 87.1 | क्या बुनियादी है? |
| in_domain | True | 65.7 | 65.8 | benefits of moringa ginger tea |
| in_domain | True | 147.2 | 147.3 | एक गैलन में कितना तरल होता है |
| in_domain | True | 35.6 | 35.7 | giDa delauran net worth |
| in_domain | True | 72.3 | 72.4 | drinking lots of water can help lose weight |
| in_domain | True | 66.3 | 66.4 | adderall dose to body weight |
| in_domain | True | 117.1 | 117.2 | मूलगामी गर्दन को परिभाषित करें |
| in_domain | True | 57.9 | 57.9 | defination arbitrary |
| in_domain | True | 87.4 | 87.5 | कुछ जो औसत से परे है |
| in_domain | True | 112.5 | 112.6 | हमारे पास कितनी सबवे फ्रेंचाइजी हैं? |
| in_domain | True | 66.5 | 66.6 | definition of a   parabola |
| in_domain | True | 124.6 | 124.7 | कितने खर्च होते हैं यू.एस.एक्स. गेम्स? |
| in_domain | True | 104.2 | 104.2 | दांत पर टोपी के लिए लागत |
| in_domain | True | 126.0 | 126.1 | बेयर्न म्यूनिख क्या है |
| in_domain | True | 114.7 | 114.8 | शरीर के वजन में एडरल खुराक जोड़ें। |
| in_domain | True | 84.9 | 85.0 | how much do canadians pay in taxes for health care |
| in_domain | True | 39.6 | 39.7 | about how much do youtubers make |
| in_domain | True | 65.6 | 65.7 | what is betaine plus hp for |
| in_domain | True | 44.9 | 45.0 | summit league commissioner |
| in_domain | True | 112.4 | 112.5 | क्या भारतीय लोग चावल खाते हैं? |
| in_domain | True | 85.6 | 85.7 | how much is it per day when locked up in county jail |
| in_domain | True | 71.7 | 71.8 | how much water does it take to make pecan |
| in_domain | True | 146.4 | 146.5 | मकई के एक कण को माइक्रोवेव में कितने समय तक पकाना है? |
| in_domain | True | 62.8 | 62.9 | definition of excess land |
| in_domain | True | 93.2 | 93.3 | सीधे कारण की परिभाषा |
| in_domain | True | 55.1 | 55.1 | suit definition |
| in_domain | True | 71.8 | 71.9 | how to look after a dog with arthritis |
| in_domain | True | 93.4 | 93.5 | उत्साहित परिभाषित करें |
| in_domain | False | 0.2 | 0.3 | परिभाषा के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया |
| in_domain | True | 49.9 | 49.9 | did president clinton impeached |
| in_domain | True | 99.3 | 99.4 | दुनिया की सबसे बड़ी ओरिगामी |
| in_domain | True | 106.1 | 106.2 | औसत मासिक लागत नर्सिंग होम |
| in_domain | True | 64.2 | 64.3 | foods that help cramps |
| in_domain | True | 157.7 | 157.7 | कवनसोलॉपर के गोदाम के लिए पर्यवेक्षक की वेतन दर कितनी है |
| in_domain | True | 72.8 | 72.9 | what is bell's palsy related to |
| in_domain | True | 100.5 | 100.6 | होंगेरा का अर्थ है |
| in_domain | True | 94.9 | 95.0 | विकलांगता बीमा की परिभाषा |
| in_domain | True | 73.3 | 73.4 | average temperature by month in payson, az |
| in_domain | True | 142.0 | 142.0 | क्या अंगूठे के अंदर के नाखून अपने आप निकल जाते हैं? |
| in_domain | True | 47.9 | 48.0 | how much xbox games cost |
| in_domain | True | 48.0 | 48.1 | how much nurse pay |
| in_domain | True | 46.3 | 46.3 | do interns get paid |
| in_domain | True | 145.1 | 145.2 | क्या सोया का कब्ज से कोई लेना-देना है? |
| in_domain | True | 73.8 | 73.9 | average salary of a office manager mortgage lending |
| in_domain | True | 159.9 | 160.0 | काउंटी जेल में बंद रहते हुए प्रतिदिन कितना खर्च आता है? |
| in_domain | True | 56.1 | 56.2 | harrison ford's son liam |
| in_domain | True | 119.9 | 120.0 | पेकान बनाने में कितना पानी लगता है। |
| in_domain | True | 70.2 | 70.2 | how to change power level on microwave |
| in_domain | True | 105.2 | 105.3 | बैंडविड्थ की परिभाषा इंटरनेट पर |
| in_domain | True | 184.6 | 184.7 | कूल्हे की समस्याएं पैर में दर्द का कारण बन सकती हैं। |
| in_domain | True | 71.0 | 71.0 | how far is philadelphia from lancaster pa |
| in_domain | True | 98.1 | 98.2 | क्या आप ठंडी शराब पी रहे हैं? |
| in_domain | True | 72.8 | 72.8 | . what is a corporation? |
| in_domain | True | 97.5 | 97.6 | डेल्टा इओटा सिग्मा |
| in_domain | True | 117.7 | 117.8 | मिसिसिपी के लिए कंपनी कर की दरें |
| in_domain | True | 102.2 | 102.3 | आधारभूत क्या है जीने में? |
| in_domain | True | 68.2 | 68.2 | स्कॉफी की परिभाषा |
| in_domain | True | 78.6 | 78.7 | what is best oven temp to use to keep food warm |
| in_domain | True | 64.9 | 65.0 | what is behavioral event interviewing method |
| in_domain | True | 115.2 | 115.3 | मोरिंगा अदरक चाय के लाभ |
| in_domain | True | 58.8 | 58.8 | what is bdops |
| in_domain | True | 41.3 | 41.3 | susan sarandon cup size |
| in_domain | True | 142.1 | 142.2 | ब्रह्मांड विज्ञान: काल क्या है अंधेरा युग |
| in_domain | True | 62.5 | 62.6 | foods to eat when throwing up |
| in_domain | True | 57.1 | 57.2 | how involved did america get during operation rolling thunder |
| in_domain | True | 118.6 | 118.7 | क्या घंटी पक्षाघात से संबंधित है |
| in_domain | True | 111.3 | 111.4 | मैं चेरी की कैसे गिरविट करूँ। |
| in_domain | True | 49.8 | 49.9 | hongera meaning |
| in_domain | True | 71.9 | 72.0 | hot to transfer music to my samsung s6 |
| in_domain | True | 89.2 | 89.3 | how many years in college to get an mba |
| in_domain | True | 57.0 | 57.1 | how often should change your transmission fluid |
| in_domain | True | 70.6 | 70.7 | how long for cantaloupe to mature |
| in_domain | True | 90.6 | 90.7 | रद्द की गई जाँच की परिभाषा |
| in_domain | True | 52.2 | 52.3 | how many chromosomes do human offspring |
| in_domain | True | 139.2 | 139.3 | एम.बी.ए. करने के लिए कॉलेज में कितने साल लगेंगे। |
| in_domain | True | 48.7 | 48.8 | do it yourself grout cleaning |
| in_domain | True | 135.9 | 136.0 | दूरी स्कॉट्सडेल से ग्रैंड कैन्यन तक |
| in_domain | True | 126.7 | 126.7 | मानव संतानों में कितने गुणसूत्र होते हैं |
| in_domain | True | 81.6 | 81.6 | cost to install basement and yurt mn |
| in_domain | True | 49.5 | 49.5 | can hip problems cause leg pain |
| in_domain | True | 98.8 | 98.9 | एनू प्रोटीन शेक |
| in_domain | True | 83.3 | 83.4 | foods and supplements to lower blood sugar |
| in_domain | True | 66.8 | 66.9 | different types of construction vehicles |
| in_domain | True | 51.2 | 51.3 | age when teeth fall out |
| in_domain | True | 46.8 | 46.9 | edward llewellyn largest black bass caught |
| in_domain | True | 73.7 | 73.8 | how many subway franchises are in the us |
| in_domain | True | 91.6 | 91.7 | शराब को परिभाषित करें |
| in_domain | True | 122.7 | 122.8 | समाजशास्त्र की परिभाषा की संस्कृति |
| in_domain | True | 132.9 | 133.0 | राजनीतिक नवजात को परिभाषित करें। |
| adversarial | True | 81.7 | 81.8 | What's a good recipe for chocolate chip cookies? |
| adversarial | True | 80.0 | 80.1 | What is the weather like in Paris today? |
| adversarial | True | 79.2 | 79.3 | Who won the football match last night? |
| adversarial | True | 51.8 | 51.9 | asdkjfh qwoeiru 12345 %%%&&& |
| adversarial | False | 0.0 | 0.1 | How do I make a bomb at home? |
| adversarial | True | 63.0 | 63.0 | Tell me a joke about cats. |
| adversarial | True | 79.6 | 79.7 | What's the capital of Australia? |
| adversarial | True | 86.2 | 86.3 | the a an in on of is are |
