import json
import os
import traceback
from PIL import Image, ImageFile, PngImagePlugin

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .interleave_datasets.interleave_t2i_dataset import InterleavedBaseIterableDataset


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class DynamicsJSONLIterableDataset(InterleavedBaseIterableDataset):
    def __init__(
        self, dataset_name, transform, vit_transform, tokenizer, 
        jsonl_path_list, prompt_path_list, data_dir_list, num_used_data, 
        local_rank=0, world_size=1, num_workers=8, data_status=None, 
        shuffle_lines=False, shuffle_seed=0,
    ):
        """
        jsonl_path_list: list of jsonl file paths
        data_dir_list: list of image directories containing the images of each jsonl file
        num_used_data: list of number of sampled data points for each jsonl
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(
            jsonl_path_list,
            prompt_path_list, 
            data_dir_list, 
            num_used_data, 
            shuffle_lines, 
            shuffle_seed,
        )
        self.set_epoch()

    def get_data_paths(
        self, 
        jsonl_path_list,
        prompt_path_list, 
        data_dir_list, 
        num_used_data, 
        shuffle_lines, 
        shuffle_seed,
    ):
        data_paths = []
        for jsonl_path, image_dir, prompt_path, num_data_point in zip(
            jsonl_path_list, data_dir_list, prompt_path_list, num_used_data
        ):
            with open(jsonl_path, 'r') as f:
                raw_data = f.readlines()
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            raw_data = raw_data[:num_data_point]
            data_paths.extend([(json_data, prompt_path, image_dir) for json_data in raw_data])
        return data_paths

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (data, prompt_path, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):

                try:
                    data_item = json.loads(data)
                    raw_images = [
                        pil_img2rgb(Image.open(os.path.join(image_dir, image)))
                        for image in data_item['images']
                    ]
                    if 'init_image' in data_item:
                        init_image = pil_img2rgb(Image.open(os.path.join(image_dir, data_item['init_image'])))
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                except:
                    traceback.print_exc()
                    continue

                # elements = self.change_format(data_item, len(image_tensor_list))
                # prompt = data_item['prompt']
                action_instructions = list(data_item['action_sequence'])
                horizon_len = len(action_instructions)
                data = self._init_data()
                data = self._add_text(
                    data,
                    prompt,
                    need_loss=False,
                    enable_cfg=True
                )
                if 'init_image' in data_item:
                    data = self._add_image(
                        data, 
                        init_image,
                        need_loss=False, 
                        need_vae=True, 
                        need_vit=True,
                        enable_cfg=True
                    )
                data = self._add_image(
                    data, 
                    raw_images[0],
                    need_loss=False, 
                    need_vae=True, 
                    need_vit=True,
                    enable_cfg=True
                )
                # horizon_len = 1
                for idx in range(horizon_len):
                    data = self._add_text(
                        data,
                        action_instructions[idx],
                        need_loss=False,
                        enable_cfg=True
                    )
                    if idx < horizon_len - 1:
                        data = self._add_image(
                            data, 
                            raw_images[idx+1],
                            need_loss=True, 
                            need_vae=True, 
                            need_vit=True,
                            enable_cfg=True
                        )
                    else:
                        data = self._add_image(
                            data, 
                            raw_images[idx+1],
                            need_loss=True, 
                            need_vae=False, 
                            need_vit=False,
                            enable_cfg=True
                        )

                yield dict(
                    image_tensor_list=data["image_tensor_list"],
                    text_ids_list=data["text_ids_list"],
                    sequence_plan=data["sequence_plan"],
                    num_tokens=data["num_tokens"],
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    }
                )

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
