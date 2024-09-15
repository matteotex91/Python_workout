import pickle
import os

file_folder = os.getcwd() + "/audio/datasets/"


def save_transferfunction(
    sampled_frequencies, fft_frequencies, fft_amplitudes, fft_sigmas, run_id: int = None
):
    if run_id is None:
        run_id = (
            max(
                [
                    int(f.split("tf")[1].split(".")[0])
                    for f in os.listdir(file_folder)
                    if os.path.isfile(file_folder + f)
                    and any(char.isdigit() for char in f)
                ]
            )
            + 1
        )
    with open(file_folder + f"tf{run_id}.pickle", "wb") as f:
        pickle.dump(
            {
                "sampled_frequencies": sampled_frequencies,
                "fft_frequencies": fft_frequencies,
                "fft_amplitudes": fft_amplitudes,
                "fft_sigmas": fft_sigmas,
            },
            f,
        )


def load_transferfunction(run_id):
    path = file_folder + f"tf{run_id}.pickle"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        datadict = pickle.load(f)
    sampled_frequencies = datadict["sampled_frequencies"]
    fft_frequencies = datadict["fft_frequencies"]
    fft_amplitudes = datadict["fft_amplitudes"]
    fft_sigmas = datadict["fft_sigmas"]
    return (sampled_frequencies, fft_frequencies, fft_amplitudes, fft_sigmas)


def load_transferfunction_old(run_id):
    path = file_folder + f"tf{run_id}.pickle"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        datadict = pickle.load(f)
    sampled_frequencies = datadict["frequencies"]
    fft_frequencies = datadict["fft_freq_vector"]
    fft_amplitudes = datadict["fft_responses"]
    fft_sigmas = datadict["fft_sigmas"]
    return (sampled_frequencies, fft_frequencies, fft_amplitudes, fft_sigmas)
