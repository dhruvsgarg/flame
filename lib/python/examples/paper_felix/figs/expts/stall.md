slot-hours = Σ over tasks; async capacity = c(t) × run time, sync = its picks' task time; a run ends at the target or 120 h

| dataset | trace | stack | run h | useful compute (slot-h) | compute discarded (slot-h) | held by departed client (slot-h) | finished, waiting (barrier) (slot-h) | idle (no task) (slot-h) | free, eligible clients await their commit (slot-h) | lost share | tasks whose client left | slot time lost per departure (s) | tasks released early | released by |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR-10 | Syn-50% | Felix | 72 | 4315 | 1 | 0 | 0 | 3 | 0 | 0% | 1.7% | 0 | 12596 | eviction |
| CIFAR-10 | Syn-50% | FedBuff | 81 | 2998 | 1 | 1678 | 0 | 185 | 0 | 38% | 1.7% | 626 | 5125 | timeout |
| CIFAR-10 | Syn-50% | FedBuff* | 81 | 3097 | 0 | 1730 | 0 | 3 | 0 | 36% | 1.7% | 626 | 5146 | timeout |
| CIFAR-10 | Syn-50% | REFL | 120 | 1005 | 1 | 107 | 1146 | 0 | 0 | 5% | 1.3% | 81 | 4749 | timeout |
| CIFAR-10 | Syn-50% | FedDance | 120 | 744 | 485 | 109 | 693 | 0 | 0 | 29% | 1.4% | 83 | 5403 | timeout |
| CIFAR-10 | Syn-50% | FedAvg | 120 | 742 | 490 | 107 | 688 | 0 | 0 | 29% | 1.4% | 83 | 5411 | timeout |
| CIFAR-10 | Syn-50% | Oort | 120 | 332 | 73 | 404 | 651 | 0 | 0 | 33% | 13.4% | 836 | 1534 | timeout |
| CIFAR-10 | MobiPerf | Felix | 120 | 3263 | 1 | 2 | 0 | 0 | 3 | 0% | 0.5% | 2 | 2441 | eviction |
| CIFAR-10 | MobiPerf | FedBuff | 120 | 5025 | 1 | 485 | 0 | 1689 | 0 | 30% | 0.3% | 378 | 2425 | timeout |
| CIFAR-10 | MobiPerf | FedBuff* | 120 | 5074 | 2 | 437 | 0 | 1485 | 202 | 27% | 0.3% | 373 | 2183 | timeout |
| CIFAR-10 | MobiPerf | REFL | 120 | 1157 | 0 | 44 | 1151 | 0 | 0 | 2% | 0.6% | 57 | 2112 | timeout |
| CIFAR-10 | MobiPerf | FedDance | 120 | 922 | 575 | 18 | 684 | 0 | 0 | 27% | 0.3% | 52 | 806 | timeout |
| CIFAR-10 | MobiPerf | FedAvg | 120 | 914 | 592 | 19 | 678 | 0 | 0 | 28% | 0.3% | 51 | 841 | timeout |
| CIFAR-10 | MobiPerf | Oort | 120 | 528 | 72 | 93 | 794 | 0 | 0 | 11% | 3.3% | 648 | 333 | timeout |
| Speech | Syn-50% | Felix | 66 | 1322 | 2 | 4 | 0 | 0 | 0 | 0% | 13.5% | 4 | 3568 | eviction |
| Speech | Syn-50% | FedBuff | 120 | 1014 | 1 | 978 | 0 | 407 | 0 | 58% | 17.9% | 1022 | 617 | timeout |
| Speech | Syn-50% | FedBuff* | 120 | 1258 | 2 | 1141 | 0 | 0 | 0 | 48% | 16.7% | 1022 | 716 | timeout |
| Speech | Syn-50% | REFL | 120 | 195 | 1 | 194 | 246 | 0 | 0 | 31% | 11.1% | 954 | 469 | timeout |
| Speech | Syn-50% | FedDance | 120 | 151 | 79 | 172 | 201 | 0 | 0 | 42% | 10.7% | 828 | 469 | timeout |
| Speech | Syn-50% | FedAvg | 120 | 145 | 77 | 168 | 200 | 0 | 0 | 42% | 10.2% | 832 | 446 | timeout |
| Speech | Syn-50% | Oort | 120 | 117 | 10 | 180 | 167 | 0 | 0 | 40% | 16.6% | 1367 | 245 | timeout |
| Speech | MobiPerf | Felix | 120 | 1224 | 2 | 11 | 0 | 18 | 273 | 2% | 3.2% | 49 | 596 | eviction |
| Speech | MobiPerf | FedBuff | 120 | 1277 | 2 | 365 | 0 | 718 | 40 | 45% | 3.8% | 1262 | 635 | timeout |
| Speech | MobiPerf | FedBuff* | 120 | 1333 | 3 | 280 | 0 | 466 | 321 | 31% | 2.9% | 1171 | 488 | timeout |
| Speech | MobiPerf | REFL | 120 | 281 | 1 | 59 | 272 | 0 | 0 | 10% | 2.8% | 645 | 224 | timeout |
| Speech | MobiPerf | FedDance | 120 | 218 | 139 | 45 | 247 | 0 | 0 | 28% | 2.4% | 555 | 186 | timeout |
| Speech | MobiPerf | FedAvg | 120 | 214 | 146 | 40 | 239 | 0 | 0 | 29% | 2.1% | 534 | 162 | timeout |
| Speech | MobiPerf | Oort | 120 | 158 | 3 | 45 | 214 | 0 | 0 | 11% | 5.1% | 1055 | 96 | timeout |
