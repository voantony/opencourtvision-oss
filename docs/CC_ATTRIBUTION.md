# CC attribution

Each public clip has a block here and a sha256 in `benchmarks/public_data/clips_manifest.json`. CC0 needs a source line even though attribution is not a legal condition. CC BY must reproduce the credit the author asked for.

Template for a new block:

```
id:
title:
author:
url:
licence: CC0 | CC BY
sha256:
accessed:
```

Media bytes are not committed. `scripts/fetch_public_clips.py` downloads them into a cache and checks the sha256. The session-01 clips are stored in the project's generic package registry (`session-01-clips/1.0.0`) and copied into the volunteer labeler on GitLab Pages at build time.

## session-01 (padel, 46 clips)

Licence checked 2026-11-09 on each YouTube watch page: the "License" row reads "Creative Commons Attribution license (reuse allowed)" (YouTube CC BY, https://www.youtube.com/t/creative_commons).

How the clips were cut: one clip per candidate point window with ffmpeg (`-ss` before `-i`, `-vf scale=-2:480`, `libx264 -preset veryfast -crf 30`), about 1 s of padding before and after the window, audio removed. The source times are in the manifest (`source_start_s`, `source_end_s`). These are changes to the original under CC BY. The clips are unlabeled. No point winner in this repository comes from them.

```
id: GKWNVYeEAlE
title: Intermediate Level 3.5+ | Padel Match | Padel Match Point - @ Mariano Sports Arena
author: Padel Match Point (https://www.youtube.com/channel/UC-HVV4JL9YIsW4BgL9w9svw)
url: https://www.youtube.com/watch?v=GKWNVYeEAlE
licence: CC BY (YouTube "Creative Commons Attribution license (reuse allowed)")
accessed: 2026-11-09
changes: cut into short clips, scaled to 480p, re-encoded, audio removed
```

```
id: 632k3fKraeA
title: Padel match level 3 - Partido de Padél - Partita padel competitiva - Full match
author: Padel Match Point (https://www.youtube.com/channel/UC-HVV4JL9YIsW4BgL9w9svw)
url: https://www.youtube.com/watch?v=632k3fKraeA
licence: CC BY (YouTube "Creative Commons Attribution license (reuse allowed)")
accessed: 2026-11-09
changes: cut into short clips, scaled to 480p, re-encoded, audio removed
```

```
id: 2p6PcKP0720
title: Bogdan/Costi VS Tudor/Ludovico - Padel FZN Arena - Level 2.5
author: Amateur Padel (https://www.youtube.com/channel/UCnY3YfSon2KM4y9awSs-Yrg)
url: https://www.youtube.com/watch?v=2p6PcKP0720
licence: CC BY (YouTube "Creative Commons Attribution license (reuse allowed)")
accessed: 2026-11-09
changes: cut into short clips, scaled to 480p, re-encoded, audio removed
```

| Clip | Source | sha256 |
|---|---|---|
| `2p6PcKP0720_1.mp4` | 2p6PcKP0720 | `697f37c467ef2d3bb3ae39e3f312fa47a499bf275731bda6e350ad6a0ac34cff` |
| `2p6PcKP0720_2.mp4` | 2p6PcKP0720 | `eb041a2983f7a8e359a65ccdedef715d8ca30c811887be1e128d2a6587fb3253` |
| `2p6PcKP0720_3.mp4` | 2p6PcKP0720 | `f31a1e16a63f0bd703df39cd91da491559103eeabc76740cb7e806b858c413b1` |
| `2p6PcKP0720_4.mp4` | 2p6PcKP0720 | `dd1889e0c3ccb89faff533b89443bf7176c4fd92bca267e5bf168fe609dac0c7` |
| `2p6PcKP0720_5.mp4` | 2p6PcKP0720 | `1546e6cd2849768c3e6ce43d688a1687e654363bf0ce25d2a6f1b79706e22422` |
| `632k3fKraeA_1.mp4` | 632k3fKraeA | `580def59c6a54e14632289bc023243004933510ab09292491e196deb7392cf10` |
| `632k3fKraeA_2.mp4` | 632k3fKraeA | `ee278bfc5650aabfc0cf5e6d0b676a7fc02c50652bd5743bdf119cd15e75ee22` |
| `632k3fKraeA_3.mp4` | 632k3fKraeA | `1e47d833842908fb42c33c4489d76dad843e7068f69037662af01de4dc062683` |
| `632k3fKraeA_4.mp4` | 632k3fKraeA | `14c371bfaa880ad3dcce57f6908496e5af1331143f8dab8fab315ca219327914` |
| `632k3fKraeA_5.mp4` | 632k3fKraeA | `97baf76d593a9ac25aa32f6eda04e95bffbe975bca2c1ef041cee188a0448086` |
| `632k3fKraeA_6.mp4` | 632k3fKraeA | `9c1de8c1272983dbbbc64b492c70ff53c22542e1972d74155be5a26d4989b259` |
| `632k3fKraeA_7.mp4` | 632k3fKraeA | `115387e9d4adb66e1489c4b48b56b9999e0d9288305c096c74d508782d6e7265` |
| `632k3fKraeA_8.mp4` | 632k3fKraeA | `f01bad09863d0c48a78bbec134d6d8e9b09729a5ecb48358e79ce2253a1af0fd` |
| `632k3fKraeA_9.mp4` | 632k3fKraeA | `909433acc2b7b927747fa1edbbbe569026c03c73385d6c3c9d839372896cbc86` |
| `632k3fKraeA_10.mp4` | 632k3fKraeA | `2ba5eebfcf87ac80135142383071d5f121f9810125d0152881448f975013040a` |
| `632k3fKraeA_11.mp4` | 632k3fKraeA | `49866b541d27feaa0b1603c6f041eeaf94b875708d8fb79e170c05a54b3fb236` |
| `632k3fKraeA_12.mp4` | 632k3fKraeA | `22a581809eb05dad450a82ee9c77f5183485c7bfe59525d59d57eeffc0e1f0f1` |
| `632k3fKraeA_13.mp4` | 632k3fKraeA | `f239d2bf9758875c936def4791ce8bd5d9467b199d09326a34c78749b21dacc7` |
| `632k3fKraeA_14.mp4` | 632k3fKraeA | `ebb623e5a7e0d989f2b64991e4c49e9d5126ddb3a808b62a477ca4c38d9d3678` |
| `GKWNVYeEAlE_1.mp4` | GKWNVYeEAlE | `90dd19d203072b2e6eff6e6fc776316b4153ceed93b37c931d58b46770dbf4b3` |
| `GKWNVYeEAlE_2.mp4` | GKWNVYeEAlE | `17b4b93a7762a7f9209a6140e12e42d84b4536ffaab048304fc820eb1ee2f7d1` |
| `GKWNVYeEAlE_3.mp4` | GKWNVYeEAlE | `5c8e1f01ee508e55d58402f8b607a9232cab3ca5126f3fd70b3e0a8fec6185b6` |
| `GKWNVYeEAlE_4.mp4` | GKWNVYeEAlE | `cee8192fb216bff26b8e2db0f70853410a9aa9af537768f2edeabc5a390dade3` |
| `GKWNVYeEAlE_5.mp4` | GKWNVYeEAlE | `c517f2f616e0205c91b92a5605ab64bfeea665dba3832a45c81ae721049946c4` |
| `GKWNVYeEAlE_6.mp4` | GKWNVYeEAlE | `2a630bd570363f1d4137df26923f8ac1daf8ff5e9b8b755761edc50dd05bab67` |
| `GKWNVYeEAlE_7.mp4` | GKWNVYeEAlE | `2ff8f87438380940b17484e04c85b4c3c0864125bf4b94ed7aac03430162eb49` |
| `GKWNVYeEAlE_8.mp4` | GKWNVYeEAlE | `fa127ee50a99bb65ce72f8679517dcca6829e660b03f69f3f3654ba4b4a75a8a` |
| `GKWNVYeEAlE_9.mp4` | GKWNVYeEAlE | `255449ba7492014a2cba1c493583db4d591653220f10e38c8f5a43c3fde7e667` |
| `GKWNVYeEAlE_10.mp4` | GKWNVYeEAlE | `f74c6959748a967ae990ff67bc13e5bb315aefe8a0b75014b50a4aa706efe5c1` |
| `GKWNVYeEAlE_11.mp4` | GKWNVYeEAlE | `1bc60f787325e17c37eecde7bd93373d3673664262b85c2936d13f988d62e30a` |
| `GKWNVYeEAlE_12.mp4` | GKWNVYeEAlE | `cf4e9f0a31834d5189d810973b1d22804d30afd48ec85d4496667be82836ce26` |
| `GKWNVYeEAlE_13.mp4` | GKWNVYeEAlE | `ef2f9a7ba2497425c2aec4eec48ca29860ae76642fad1fcbba0b98f2892400cc` |
| `GKWNVYeEAlE_14.mp4` | GKWNVYeEAlE | `b9fb0d78289c13bc906e0c26a54da5af0b96ec91102a29c0d57ee6a01c68f808` |
| `GKWNVYeEAlE_15.mp4` | GKWNVYeEAlE | `5d8ad085920609d1edf32d1a9a06957973547e5e07597ff2277a644f0a06de5d` |
| `GKWNVYeEAlE_16.mp4` | GKWNVYeEAlE | `c8a5e21ecd74eab89149e818f7420312e35f2a20fae5fa751d08501c38f25ed3` |
| `GKWNVYeEAlE_17.mp4` | GKWNVYeEAlE | `2051efb49b9962064cec7db383b6675c5ee5e12019a935fc8377f4b903f6949f` |
| `GKWNVYeEAlE_18.mp4` | GKWNVYeEAlE | `bb86fde4a38b283145d0a4e9014389739f76c0bddd29689d71d5ab5b55865179` |
| `GKWNVYeEAlE_19.mp4` | GKWNVYeEAlE | `28cf768f7e252d3e6abe15b793f9a9722ed6fad9702f78ce32f0bff07da5af2c` |
| `GKWNVYeEAlE_20.mp4` | GKWNVYeEAlE | `77cf07a605a519d9fd32f88fd76a40b83b995ee14b7155343e03c597aa021444` |
| `GKWNVYeEAlE_21.mp4` | GKWNVYeEAlE | `5e8f457ae7d857ee171c6f85ef96f204389fa32bba86135b90b553565a68c4a0` |
| `GKWNVYeEAlE_22.mp4` | GKWNVYeEAlE | `36184c2b72ed408d99cf7e2bca7d4661e57935dd19eb5260223c8fb99624156d` |
| `GKWNVYeEAlE_23.mp4` | GKWNVYeEAlE | `a28e6a860d448e2f67452fa58a3ac035f6063da08a8ba0d89ce491f5df1b742a` |
| `GKWNVYeEAlE_24.mp4` | GKWNVYeEAlE | `a1671378fee1c780a34689381160407d5f6aa919c7921b5002cedce8f8487a59` |
| `GKWNVYeEAlE_25.mp4` | GKWNVYeEAlE | `dcf8ab9620a4cc3d7754b496765ce6efc17300650a2d87adeffed9b1e1af3c36` |
| `GKWNVYeEAlE_26.mp4` | GKWNVYeEAlE | `49b6e8e9f1240883a1a94ffd70214efc4a4212fc0589fc9fd30d083f3a6ab395` |
| `GKWNVYeEAlE_27.mp4` | GKWNVYeEAlE | `0d8939ecbfebe0fd7e9aa864f40ea25b7e76f90b62fba99cd312b901d7cae2ca` |
