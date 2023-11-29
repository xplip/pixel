from huggingface_hub import login, HfApi, snapshot_download
import os
login(token="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")
api = HfApi()
# api.create_repo(repo_id="pixel-model-groupproject")

# api.upload_folder(
#     folder_path="/exports/eddie/scratch/s2522559/pixel_project/experiments/nov14-pretrain1/checkpoint-50000",
#     path_in_repo="pixel-replic/checkpoint-50000",
#     repo_id="yiwang454/pixel-model-groupproject",
#     repo_type="model",
# )

# api.upload_folder(
#     folder_path="/exports/eddie/scratch/s2522559/pixel_project/experiments/nov14-pretrain1/checkpoint-10000",
#     path_in_repo="pixel-replic/checkpoint-10000",
#     repo_id="yiwang454/pixel-model-groupproject",
#     repo_type="model",
# )
if not os.path.exists("/mnt/ceph_rbd/pixel_project/experiments/nov14-pretrain1"):
    os.mkdir("/mnt/ceph_rbd/pixel_project/experiments/nov14-pretrain1")
snapshot_download(repo_id="yiwang454/pixel-model-groupproject", 
                repo_type="model",
                local_dir="/mnt/ceph_rbd/pixel_project/experiments/nov14-pretrain1")