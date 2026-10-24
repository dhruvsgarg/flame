"""FX-N115: task timeout = 1.5 x the baseline's longest modeled task over the reference split."""
from flame.launch import task_timeout as T

CIFAR = dict(model_kbit=16800.3125, reference_split="cifar10_alpha0.1_n300")
SPEECH = dict(model_kbit=226665.34375, reference_split="google_speech_alpha0.1_n100")


def test_epoch_baselines_match_the_longest_fedscale_epoch():
    assert round(max(T.task_times_s("fedbuff", "cifar10", CIFAR))) == 451  # FX-D197's 903 s = 2 x this
    assert round(max(T.task_times_s("felix", "google_speech", SPEECH))) == 1280
    assert T.task_timeout_s("felix", "cifar10", CIFAR) == 1.5 * max(T.task_times_s("felix", "cifar10", CIFAR))


def test_timeout_follows_local_work():
    oort, refl = T.task_timeout_s("oort", "cifar10", CIFAR), T.task_timeout_s("refl", "cifar10", CIFAR)
    assert refl < oort < T.task_timeout_s("fedbuff", "cifar10", CIFAR)  # 10 samples < 600 < one epoch (big shards)
    assert len(T.task_times_s("refl", "google_speech", SPEECH)) == 100
