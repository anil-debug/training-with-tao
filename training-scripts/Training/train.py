import os
import subprocess

class TAOTrainingFlow:
    def __init__(self, tao_with_meta_path):
        self.tao_with_meta_path = tao_with_meta_path

    def setup_environment(self):
        os.environ["KEY"] = "nvidia_tlt"
        os.environ["CLI"] = "ngccli_cat_linux.zip"
        os.environ["LOCAL_PROJECT_DIR"] = os.path.join(self.tao_with_meta_path, "tao-experiments")
        os.environ["HOST_DATA_DIR"] = os.path.join(os.environ["LOCAL_PROJECT_DIR"], "data")
        os.environ["HOST_RESULTS_DIR"] = os.path.join(os.environ["LOCAL_PROJECT_DIR"], "dino", "results")
        os.environ["HOST_SPECS_DIR"] = os.path.join(os.getcwd(), "dino", "specs")

        # Create necessary directories
        for dir_path in [os.environ["HOST_DATA_DIR"], os.environ["HOST_RESULTS_DIR"], os.environ["HOST_SPECS_DIR"]]:
            os.makedirs(dir_path, exist_ok=True)

    def download_ngc_cli(self):
        os.makedirs(os.path.join(os.environ["LOCAL_PROJECT_DIR"], "ngccli"), exist_ok=True)
        subprocess.run(["wget", f"https://ngc.nvidia.com/downloads/{os.environ['CLI']}", "-P", os.path.join(os.environ["LOCAL_PROJECT_DIR"], "ngccli")])
        subprocess.run(["unzip", "-u", os.path.join(os.environ["LOCAL_PROJECT_DIR"], "ngccli", os.environ["CLI"]), "-d", os.path.join(os.environ["LOCAL_PROJECT_DIR"], "ngccli")])
        os.remove(os.path.join(os.environ["LOCAL_PROJECT_DIR"], "ngccli", os.environ["CLI"]))
        os.environ["PATH"] = os.path.join(os.environ["LOCAL_PROJECT_DIR"], "ngccli", "ngc-cli") + ":" + os.environ["PATH"]

    def list_pretrained_models(self):
        subprocess.run(["ngc", "registry", "model", "list", "nvidia/tao/pretrained_dino_nvimagenet:*"])

    def download_pretrained_model(self):
        subprocess.run(["ngc", "registry", "model", "download-version", "nvidia/tao/pretrained_dino_nvimagenet:fan_base_hybrid_nvimagenet", "--dest", os.path.join(os.environ["LOCAL_PROJECT_DIR"], "dino")])

    def train_dino_model(self):
        subprocess.run(["dino", "train", "-e", os.path.join(os.environ["HOST_SPECS_DIR"], "train.yaml"), "-r", os.environ["HOST_RESULTS_DIR"]])

    def rename_trained_model(self):
        epoch = "098"
        checkpoint = subprocess.check_output(["ls", os.path.join(os.environ["HOST_RESULTS_DIR"], "train", "*.pth"), "|", "grep", f"epoch={epoch}", "|", "head", "-n", "1"]).decode().strip()
        subprocess.run(["cp", checkpoint, os.path.join(os.environ["HOST_RESULTS_DIR"], "train", "dino_model.pth")])

    def evaluate_model(self):
        checkpoint = os.path.join(os.environ["HOST_RESULTS_DIR"], "train", "dino_model.pth")
        subprocess.run(["dino", "evaluate", "-e", os.path.join(os.environ["HOST_SPECS_DIR"], "evaluate.yaml"), "evaluate.checkpoint=" + checkpoint, "results_dir=" + os.environ["HOST_RESULTS_DIR"]])

    def infer_model(self):
        checkpoint = os.path.join(os.environ["HOST_RESULTS_DIR"], "train", "dino_model.pth")
        subprocess.run(["dino", "inference", "-e", os.path.join(os.environ["HOST_SPECS_DIR"], "infer.yaml"), "inference.checkpoint=" + checkpoint, "results_dir=" + os.environ["HOST_RESULTS_DIR"]])

if __name__ == "__main__":
    tao_with_meta_path = "./tao-experiments"
    tao_training_flow = TAOTrainingFlow(tao_with_meta_path)
    tao_training_flow.setup_environment()
    tao_training_flow.download_ngc_cli()
    tao_training_flow.list_pretrained_models()
    tao_training_flow.download_pretrained_model()
    tao_training_flow.train_dino_model()
    tao_training_flow.rename_trained_model()
    tao_training_flow.evaluate_model()
    tao_training_flow.infer_model()
