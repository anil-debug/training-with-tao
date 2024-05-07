# import os
# import subprocess

# class TAOTrainingFlow:
#     """
#     Initialize TAOTrainingFlow object.

#     Args:
#         tao_with_meta_path (str): Path to the TAO experiments folder.
#         cli_name (str): Name of the NGC CLI ZIP file.
#         pretrained_model (str): Name of the pretrained model.
#         epochs (str): Epochs to use for training.
#     """
#     def __init__(self, tao_with_meta_path, cli_name="ngccli_cat_linux.zip", pretrained_model="fan_base_hybrid_nvimagenet", specs_directory='', epochs="098"):
#         self.tao_with_meta_path = tao_with_meta_path
#         self.cli_name = cli_name
#         self.pretrained_model = pretrained_model
#         self.specs_directory = specs_directory
#         self.epochs = epochs

#     def setup_environment(self):
#         """
#         Set up environment variables and create necessary directories.
#         """
#         os.environ["KEY"] = "nvidia_tlt"
#         os.environ["CLI"] = self.cli_name
#         os.environ["SPECS_DIR"] = self.specs_directory
#         os.environ["LOCAL_PROJECT_DIR"] = self.tao_with_meta_path
#         os.environ["HOST_DATA_DIR"] = os.path.join(os.environ["LOCAL_PROJECT_DIR"], "data")
#         os.environ["HOST_RESULTS_DIR"] = os.path.join(os.environ["LOCAL_PROJECT_DIR"], "dino", "results")
#         os.environ["HOST_SPECS_DIR"] = os.path.join(os.environ["SPECS_DIR"], "dino", "specs")

#         # Check if directories exist inside the tao-experiments folder, create them if not
#         for dir_path in [os.environ["HOST_DATA_DIR"], os.environ["HOST_RESULTS_DIR"], os.environ["HOST_SPECS_DIR"]]:
#             if not os.path.exists(dir_path):
#                 os.makedirs(dir_path)

#     def download_ngc_cli(self):
#         """
#         Download NGC CLI if not already downloaded.
#         """
#         ngc_cli_dir = os.path.join(os.environ["LOCAL_PROJECT_DIR"], "ngccli")
#         ngc_cli_zip_path = os.path.join(ngc_cli_dir, os.environ["CLI"])

#         # Check if NGC CLI ZIP file exists
#         if not os.path.exists(ngc_cli_zip_path):
#             os.makedirs(ngc_cli_dir, exist_ok=True)
#             subprocess.run(["wget", f"https://ngc.nvidia.com/downloads/{os.environ['CLI']}", "-P", ngc_cli_dir])
#             subprocess.run(["unzip", "-u", ngc_cli_zip_path, "-d", ngc_cli_dir])
#             os.remove(ngc_cli_zip_path)
#             os.environ["PATH"] = os.path.join(ngc_cli_dir, "ngc-cli") + ":" + os.environ["PATH"]
#             print("NGC CLI downloaded successfully.")
#         else:
#             print("NGC CLI is already downloaded.")

#     def list_pretrained_models(self, pattern="*"):
#         """
#         List pretrained models available in NGC registry.

#         Args:
#             pattern (str): Pattern to filter pretrained models.
#         """
#         subprocess.run(["ngc", "registry", "model", "list", f"nvidia/tao/pretrained_dino_nvimagenet:{pattern}"])

#     def download_pretrained_model(self):
#         """
#         Download pretrained model.
#         """
#         subprocess.run(["ngc", "registry", "model", "download-version", f"nvidia/tao/pretrained_dino_nvimagenet:{self.pretrained_model}", "--dest", os.path.join(os.environ["LOCAL_PROJECT_DIR"], "dino")])
   
#     def train_dino_model(self):
#         """
#         Train DINO model.
#         """
#         subprocess.run(["dino", "train", "-e", os.path.join(os.environ["HOST_SPECS_DIR"], "train.yaml"), "-r", os.environ["HOST_RESULTS_DIR"]])

#     def rename_trained_model(self):
#         """
#         Rename trained model.
#         """
#         checkpoint = subprocess.check_output(["ls", os.path.join(os.environ["HOST_RESULTS_DIR"], "train", "*.pth"), "|", "grep", f"epoch={self.epochs}", "|", "head", "-n", "1"]).decode().strip()
#         subprocess.run(["cp", checkpoint, os.path.join(os.environ["HOST_RESULTS_DIR"], "train", "dino_model.pth")])

#     def evaluate_model(self):
#         """
#         Evaluate trained model.
#         """
#         checkpoint = os.path.join(os.environ["HOST_RESULTS_DIR"], "train", "dino_model.pth")
#         subprocess.run(["dino", "evaluate", "-e", os.path.join(os.environ["HOST_SPECS_DIR"], "evaluate.yaml"), "evaluate.checkpoint=" + checkpoint, "results_dir=" + os.environ["HOST_RESULTS_DIR"]])

#     def infer_model(self):
#         """
#         Infer with trained model.
#         """
#         checkpoint = os.path.join(os.environ["HOST_RESULTS_DIR"], "train", "dino_model.pth")
#         subprocess.run(["dino", "inference", "-e", os.path.join(os.environ["HOST_SPECS_DIR"], "infer.yaml"), "inference.checkpoint=" + checkpoint, "results_dir=" + os.environ["HOST_RESULTS_DIR"]])

# if __name__ == "__main__":
#     tao_with_meta_path = "/opt/nvidia/tools/tao-experiments"  # Provide the path to the TAO experiments folder
#     cli_name = "ngccli_cat_linux.zip"  # Provide the NGC CLI ZIP file name
#     pretrained_model = "fan_base_hybrid_nvimagenet"  # Provide the name of the pretrained model
#     specs_directory = "/opt/nvidia/tools/training-scripts/Training"
#     epochs = "100"  # Provide the epochs to use for training

#     tao_training_flow = TAOTrainingFlow(tao_with_meta_path, cli_name, pretrained_model, specs_directory, epochs)
#     tao_training_flow.setup_environment()
#     tao_training_flow.download_ngc_cli()
#     tao_training_flow.list_pretrained_models()  # You can provide a pattern argument here if needed
#     tao_training_flow.download_pretrained_model()
#     tao_training_flow.train_dino_model()
#     tao_training_flow.rename_trained_model()
#     tao_training_flow.evaluate_model()
#     tao_training_flow.infer_model()
import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO)

class TAOTrainingFlow:
    """
    Initialize TAOTrainingFlow object.

    Args:
        tao_with_meta_path (str): Path to the TAO experiments folder.
        cli_name (str): Name of the NGC CLI ZIP file.
        pretrained_model (str): Name of the pretrained model.
        epochs (str): Epochs to use for training.
    """
    def __init__(self, tao_with_meta_path, cli_name="ngccli_cat_linux.zip", pretrained_model="fan_base_hybrid_nvimagenet", specs_directory='', epochs="098"):
        self.tao_with_meta_path = tao_with_meta_path
        self.cli_name = cli_name
        self.pretrained_model = pretrained_model
        self.specs_directory = specs_directory
        self.epochs = epochs

    def setup_environment(self):
        """
        Set up environment variables and create necessary directories.
        """
        os.environ["KEY"] = "nvidia_tlt"
        os.environ["CLI"] = self.cli_name
        os.environ["SPECS_DIR"] = self.specs_directory
        os.environ["LOCAL_PROJECT_DIR"] = self.tao_with_meta_path
        os.environ["HOST_DATA_DIR"] = os.path.join(os.environ["LOCAL_PROJECT_DIR"], "data")
        os.environ["HOST_RESULTS_DIR"] = os.path.join(os.environ["LOCAL_PROJECT_DIR"], "dino", "results")
        os.environ["HOST_SPECS_DIR"] = os.path.join(os.environ["SPECS_DIR"], "dino", "specs")

        # Check if directories exist inside the tao-experiments folder, create them if not
        for dir_path in [os.environ["HOST_DATA_DIR"], os.environ["HOST_RESULTS_DIR"], os.environ["HOST_SPECS_DIR"]]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def download_ngc_cli(self):
        """
        Download NGC CLI if not already downloaded.
        """
        ngc_cli_dir = os.path.join(os.environ["LOCAL_PROJECT_DIR"], "ngccli")
        ngc_cli_zip_path = os.path.join(ngc_cli_dir, os.environ["CLI"])

        # Check if NGC CLI ZIP file exists
        if not os.path.exists(ngc_cli_zip_path):
            os.makedirs(ngc_cli_dir, exist_ok=True)
            try:
                subprocess.run(["wget", f"https://ngc.nvidia.com/downloads/{os.environ['CLI']}", "-P", ngc_cli_dir], check=True)
                subprocess.run(["unzip", "-u", ngc_cli_zip_path, "-d", ngc_cli_dir], check=True)
                os.remove(ngc_cli_zip_path)
                os.environ["PATH"] = os.path.join(ngc_cli_dir, "ngc-cli") + ":" + os.environ["PATH"]
                logging.info("NGC CLI downloaded successfully.")
            except subprocess.CalledProcessError as e:
                logging.error(f"Error downloading NGC CLI: {e}")
        else:
            logging.info("NGC CLI is already downloaded.")

    def list_pretrained_models(self, pattern="*"):
        """
        List pretrained models available in NGC registry.

        Args:
            pattern (str): Pattern to filter pretrained models.
        """
        try:
            subprocess.run(["ngc", "registry", "model", "list", f"nvidia/tao/pretrained_dino_nvimagenet:{pattern}"], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error listing pretrained models: {e}")

    def download_pretrained_model(self):
        """
        Download pretrained model.
        """
        try:
            subprocess.run(["ngc", "registry", "model", "download-version", f"nvidia/tao/pretrained_dino_nvimagenet:{self.pretrained_model}", "--dest", os.path.join(os.environ["LOCAL_PROJECT_DIR"], "dino")], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error downloading pretrained model: {e}")
   
    def train_dino_model(self):
        """
        Train DINO model.
        """
        try:
            subprocess.run(["dino", "train", "-e", os.path.join(os.environ["HOST_SPECS_DIR"], "train.yaml"), "-r", os.environ["HOST_RESULTS_DIR"]], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error training DINO model: {e}")

    def rename_trained_model(self):
        """
        Rename trained model.
        """
        try:
            checkpoint = subprocess.check_output(["ls", os.path.join(os.environ["HOST_RESULTS_DIR"], "train", "*.pth"), "|", "grep", f"epoch={self.epochs}", "|", "head", "-n", "1"]).decode().strip()
            subprocess.run(["cp", checkpoint, os.path.join(os.environ["HOST_RESULTS_DIR"], "train", "dino_model.pth")], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error renaming trained model: {e}")

    def evaluate_model(self):
        """
        Evaluate trained model.
        """
        try:
            checkpoint = os.path.join(os.environ["HOST_RESULTS_DIR"], "train", "dino_model.pth")
            subprocess.run(["dino", "evaluate", "-e", os.path.join(os.environ["HOST_SPECS_DIR"], "evaluate.yaml"), "evaluate.checkpoint=" + checkpoint, "results_dir=" + os.environ["HOST_RESULTS_DIR"]], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error evaluating trained model: {e}")

    def infer_model(self):
        """
        Infer with trained model.
        """
        try:
            checkpoint = os.path.join(os.environ["HOST_RESULTS_DIR"], "train", "dino_model.pth")
            subprocess.run(["dino", "inference", "-e", os.path.join(os.environ["HOST_SPECS_DIR"], "infer.yaml"), "inference.checkpoint=" + checkpoint, "results_dir=" + os.environ["HOST_RESULTS_DIR"]], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error inferring with trained model: {e}")

if __name__ == "__main__":
    tao_with_meta_path = "/opt/nvidia/tools/tao-experiments"  # Provide the path to the TAO experiments folder
    cli_name = "ngccli_cat_linux.zip"  # Provide the NGC CLI ZIP file name
    pretrained_model = "fan_base_hybrid_nvimagenet"  # Provide the name of the pretrained model
    specs_directory = "/opt/nvidia/tools/training-scripts/Training"
    epochs = "100"  # Provide the epochs to use for training

    tao_training_flow = TAOTrainingFlow(tao_with_meta_path, cli_name, pretrained_model, specs_directory, epochs)
    tao_training_flow.setup_environment()
    tao_training_flow.download_ngc_cli()
    tao_training_flow.list_pretrained_models()  # You can provide a pattern argument here if needed
    tao_training_flow.download_pretrained_model()
    tao_training_flow.train_dino_model()
    tao_training_flow.rename_trained_model()
    tao_training_flow.evaluate_model()
    tao_training_flow.infer_model()
