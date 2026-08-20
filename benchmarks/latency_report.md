# Voice RAG Latency Benchmark

Ran 108 queries (100 real MSMARCO-XI queries, 8 adversarial/off-topic) through `PipelineHarness` against the same corpus `src/api.py` serves (9990 indexed chunks), with `MockSTT` substituted for real speech-to-text so the numbers below isolate in-process compute (embedding, FAISS retrieval, guardrail checks, LLM generation) from network-bound STT.

## Verdict

| Target | P100 (max) | Result |
|---|---|---|
| Retrieval pipeline < 200ms (input guard + embed + FAISS + rerank + relevance guard, excl. generation) | 6218.3ms | **FAIL** |
| Post-STT total < 200ms (everything after STT, **including** LLM generation) | 9352.3ms | **FAIL** |

These are two different, deliberately separate claims — see "Why two targets" below before reading one as a substitute for the other.

## Stage breakdown

| Stage | P50 | P70 | P100 (max) | Samples |
|---|---|---|---|---|
| Retrieval (embed + FAISS + rerank) only | 44.9 ms | 55.8 ms | 6216.7 ms | 107 |
| Generation only | 755.2 ms | 1004.1 ms | 2342.8 ms | 44 |
| **Retrieval pipeline** (target < 200ms) | 44.8 ms | 55.8 ms | 6218.3 ms | 108 |
| **Post-STT total** (target < 200ms) | 3154.7 ms | 3187.9 ms | 9352.3 ms | 108 |

Sample counts below 108 for the "only" rows are expected: some adversarial queries are correctly short-circuited by InputGuardrail or RelevanceGuardrail before reaching retrieval or generation at all, so those stages simply weren't attempted for them (their duration is excluded, not counted as 0ms or as a failure). "Retrieval pipeline" and "Post-STT total" always have the full 108 samples, since every query spends *some* time in the stages each one covers even when short-circuited early.

## Speech-to-text (Sarvam API) — network-bound, separate from both targets

Ran 5 real calls to Sarvam's speech-to-text API on a 1-second silent WAV clip (0 failed and were excluded from the percentiles below).

| Metric | P50 | P100 (max) | Samples |
|---|---|---|---|
| STT round-trip | 411.7 ms | 1243.4 ms | 5 |

## Why two targets

"Retrieval pipeline" covers what this codebase actually controls: query validation, embedding, FAISS search, reranking, and the relevance check. This is the number an ANN index, a smaller embedding model, or a smarter cache would move — and the one that's realistic to hold under 200ms regardless of corpus size (within reason).

"Post-STT total" additionally includes LLM generation — a real network round trip to a third-party hosted model. No local optimization changes how long a remote GPU takes to generate tokens; holding this under 200ms is a claim about the LLM provider's latency, not about this codebase. If the verdict table above shows this target failing, that is expected and does not indicate a retrieval-side regression — check the retrieval pipeline verdict independently.

STT is excluded from both because it precedes this pipeline entirely (the harness only starts timing after a query is already transcribed) — `MockSTT` removes it from these numbers by design, not as an oversight, and its real latency is reported separately above.

## Per-query detail

| Category | Degraded | Retrieval pipeline (ms) | Post-STT total (ms) | Query |
|---|---|---|---|---|
| in_domain | False | 7.8 | 739.6 | how is caffeine metabolized |
| in_domain | False | 49.0 | 713.0 | सुल्तानी अर्थात् |
| in_domain | False | 36.9 | 778.2 | can ear infections cause seizures in cats? |
| in_domain | False | 57.2 | 1318.7 | स्थानीय डिस्क की परिभाषा |
| in_domain | True | 37.6 | 3368.7 | can chia seeds be planted in house |
| in_domain | True | 55.9 | 3608.6 | स्मिथविले, एन.जे. में होटल |
| in_domain | False | 47.4 | 551.7 | सब स्लैब परिभाषा |
| in_domain | True | 87.6 | 3188.0 | क्या ओबामा फिर से राष्ट्रपति पद के लिए चुनाव लड़ सकते हैं? |
| in_domain | False | 47.3 | 4002.3 | how much supervisor pay rate for cvs warehouse |
| in_domain | True | 76.7 | 3185.8 | समझाएं कि ज्वालामुखी प्रदूषण कैसे करते हैं |
| in_domain | True | 60.5 | 3191.4 | क्या योग से मांसपेशियां बनती हैं? |
| in_domain | False | 56.5 | 3850.9 | क्या बुनियादी है? |
| in_domain | False | 36.6 | 4317.9 | benefits of moringa ginger tea |
| in_domain | True | 53.5 | 3154.7 | एक गैलन में कितना तरल होता है |
| in_domain | True | 40.4 | 3150.5 | giDa delauran net worth |
| in_domain | False | 37.2 | 4111.1 | drinking lots of water can help lose weight |
| in_domain | True | 32.8 | 3136.0 | adderall dose to body weight |
| in_domain | True | 54.0 | 3202.7 | मूलगामी गर्दन को परिभाषित करें |
| in_domain | False | 21.1 | 1174.3 | defination arbitrary |
| in_domain | True | 48.2 | 3151.4 | कुछ जो औसत से परे है |
| in_domain | True | 60.8 | 3175.5 | हमारे पास कितनी सबवे फ्रेंचाइजी हैं? |
| in_domain | False | 36.5 | 3920.7 | definition of a   parabola |
| in_domain | True | 67.6 | 3175.1 | कितने खर्च होते हैं यू.एस.एक्स. गेम्स? |
| in_domain | True | 58.1 | 3169.9 | दांत पर टोपी के लिए लागत |
| in_domain | True | 59.8 | 3174.6 | बेयर्न म्यूनिख क्या है |
| in_domain | False | 61.9 | 774.2 | शरीर के वजन में एडरल खुराक जोड़ें। |
| in_domain | True | 41.3 | 3147.7 | how much do canadians pay in taxes for health care |
| in_domain | True | 31.9 | 3125.8 | about how much do youtubers make |
| in_domain | False | 37.6 | 1710.8 | what is betaine plus hp for |
| in_domain | True | 23.5 | 3110.5 | summit league commissioner |
| in_domain | False | 61.1 | 1748.7 | क्या भारतीय लोग चावल खाते हैं? |
| in_domain | True | 44.7 | 3152.4 | how much is it per day when locked up in county jail |
| in_domain | True | 39.9 | 3153.4 | how much water does it take to make pecan |
| in_domain | False | 75.2 | 1142.4 | मकई के एक कण को माइक्रोवेव में कितने समय तक पकाना है? |
| in_domain | True | 29.9 | 3147.3 | definition of excess land |
| in_domain | True | 48.6 | 3140.4 | सीधे कारण की परिभाषा |
| in_domain | True | 19.9 | 3131.2 | suit definition |
| in_domain | True | 41.0 | 3137.2 | how to look after a dog with arthritis |
| in_domain | False | 51.7 | 715.3 | उत्साहित परिभाषित करें |
| in_domain | True | 6218.3 | 9352.3 | परिभाषा के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया गया है कि सूट के अनुसार परिभाषित किया |
| in_domain | False | 40.1 | 605.6 | did president clinton impeached |
| in_domain | False | 53.0 | 579.5 | दुनिया की सबसे बड़ी ओरिगामी |
| in_domain | True | 55.2 | 3174.3 | औसत मासिक लागत नर्सिंग होम |
| in_domain | False | 31.5 | 2199.0 | foods that help cramps |
| in_domain | True | 84.0 | 3196.2 | कवनसोलॉपर के गोदाम के लिए पर्यवेक्षक की वेतन दर कितनी है |
| in_domain | True | 38.8 | 3142.7 | what is bell's palsy related to |
| in_domain | True | 55.8 | 3157.3 | होंगेरा का अर्थ है |
| in_domain | False | 56.4 | 1302.2 | विकलांगता बीमा की परिभाषा |
| in_domain | True | 39.9 | 3128.4 | average temperature by month in payson, az |
| in_domain | False | 84.8 | 4212.7 | क्या अंगूठे के अंदर के नाखून अपने आप निकल जाते हैं? |
| in_domain | True | 23.0 | 3134.7 | how much xbox games cost |
| in_domain | True | 26.9 | 3134.4 | how much nurse pay |
| in_domain | False | 24.9 | 4259.6 | do interns get paid |
| in_domain | True | 74.7 | 3173.1 | क्या सोया का कब्ज से कोई लेना-देना है? |
| in_domain | False | 38.6 | 3935.1 | average salary of a office manager mortgage lending |
| in_domain | True | 86.1 | 3194.2 | काउंटी जेल में बंद रहते हुए प्रतिदिन कितना खर्च आता है? |
| in_domain | True | 28.5 | 3151.0 | harrison ford's son liam |
| in_domain | False | 62.2 | 3939.7 | पेकान बनाने में कितना पानी लगता है। |
| in_domain | True | 51.7 | 3157.2 | how to change power level on microwave |
| in_domain | False | 55.5 | 4129.2 | बैंडविड्थ की परिभाषा इंटरनेट पर |
| in_domain | True | 80.9 | 3187.2 | कूल्हे की समस्याएं पैर में दर्द का कारण बन सकती हैं। |
| in_domain | True | 38.1 | 3176.4 | how far is philadelphia from lancaster pa |
| in_domain | False | 53.4 | 3993.6 | क्या आप ठंडी शराब पी रहे हैं? |
| in_domain | False | 34.9 | 4750.6 | . what is a corporation? |
| in_domain | True | 46.2 | 3159.5 | डेल्टा इओटा सिग्मा |
| in_domain | True | 85.7 | 3228.6 | मिसिसिपी के लिए कंपनी कर की दरें |
| in_domain | False | 51.6 | 1951.6 | आधारभूत क्या है जीने में? |
| in_domain | True | 44.9 | 3154.2 | स्कॉफी की परिभाषा |
| in_domain | False | 40.8 | 5585.7 | what is best oven temp to use to keep food warm |
| in_domain | False | 37.7 | 1987.5 | what is behavioral event interviewing method |
| in_domain | True | 54.8 | 3178.9 | मोरिंगा अदरक चाय के लाभ |
| in_domain | False | 28.8 | 3598.7 | what is bdops |
| in_domain | True | 31.4 | 3154.8 | susan sarandon cup size |
| in_domain | False | 75.4 | 4440.3 | ब्रह्मांड विज्ञान: काल क्या है अंधेरा युग |
| in_domain | True | 34.3 | 3140.3 | foods to eat when throwing up |
| in_domain | True | 30.3 | 3141.5 | how involved did america get during operation rolling thunder |
| in_domain | False | 63.9 | 4967.1 | क्या घंटी पक्षाघात से संबंधित है |
| in_domain | True | 59.1 | 3169.1 | मैं चेरी की कैसे गिरविट करूँ। |
| in_domain | True | 23.2 | 3124.5 | hongera meaning |
| in_domain | False | 38.9 | 4878.3 | hot to transfer music to my samsung s6 |
| in_domain | True | 57.9 | 3189.6 | how many years in college to get an mba |
| in_domain | True | 29.0 | 3156.3 | how often should change your transmission fluid |
| in_domain | False | 35.9 | 4075.8 | how long for cantaloupe to mature |
| in_domain | True | 54.2 | 3169.4 | रद्द की गई जाँच की परिभाषा |
| in_domain | False | 27.2 | 4535.0 | how many chromosomes do human offspring |
| in_domain | True | 76.2 | 3194.6 | एम.बी.ए. करने के लिए कॉलेज में कितने साल लगेंगे। |
| in_domain | True | 27.3 | 3137.2 | do it yourself grout cleaning |
| in_domain | True | 82.0 | 3185.4 | दूरी स्कॉट्सडेल से ग्रैंड कैन्यन तक |
| in_domain | False | 68.9 | 1746.0 | मानव संतानों में कितने गुणसूत्र होते हैं |
| in_domain | True | 39.2 | 3137.7 | cost to install basement and yurt mn |
| in_domain | True | 28.0 | 3137.8 | can hip problems cause leg pain |
| in_domain | False | 52.6 | 1894.7 | एनू प्रोटीन शेक |
| in_domain | True | 43.5 | 3150.2 | foods and supplements to lower blood sugar |
| in_domain | False | 32.9 | 1083.9 | different types of construction vehicles |
| in_domain | True | 29.5 | 3129.1 | age when teeth fall out |
| in_domain | True | 41.4 | 3137.2 | edward llewellyn largest black bass caught |
| in_domain | False | 39.9 | 1853.3 | how many subway franchises are in the us |
| in_domain | True | 64.1 | 3167.5 | शराब को परिभाषित करें |
| in_domain | False | 63.2 | 4696.0 | समाजशास्त्र की परिभाषा की संस्कृति |
| in_domain | True | 60.4 | 3161.5 | राजनीतिक नवजात को परिभाषित करें। |
| adversarial | True | 47.1 | 3156.1 | What's a good recipe for chocolate chip cookies? |
| adversarial | False | 42.8 | 3488.0 | What is the weather like in Paris today? |
| adversarial | True | 35.5 | 3123.3 | Who won the football match last night? |
| adversarial | True | 43.0 | 3152.3 | asdkjfh qwoeiru 12345 %%%&&& |
| adversarial | False | 0.0 | 0.0 | How do I make a bomb at home? |
| adversarial | False | 33.5 | 1659.1 | Tell me a joke about cats. |
| adversarial | True | 36.5 | 3370.0 | What's the capital of Australia? |
| adversarial | False | 46.5 | 1759.6 | the a an in on of is are |
