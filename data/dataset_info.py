# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# from .interleave_datasets import UnifiedEditIterableDataset
# from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .dynamics_dataset import DynamicsJSONLIterableDataset


DATASET_REGISTRY = {
    'vlm_sft': SftJSONLIterableDataset,
    'dynamics_sft': DynamicsJSONLIterableDataset
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': 'your_data_path/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
		},
    },
    'vlm_sft': {
        'frozen_lake_reward_100_trajs': {
			'data_dir': '/scratch/s/sunyh/data/frozen_lake/vlm_sft_images',
			'jsonl_path': '/scratch/s/sunyh/data/frozen_lake/reward_100.jsonl',
			'prompt_path': '/scratch/s/sunyh/data/frozen_lake/reward_prompt.txt',
			'num_total_samples': 802
		},
        'frozen_lake_reward_250_trajs': {
			'data_dir': '/scratch/s/sunyh/data/frozen_lake/vlm_sft_images',
			'jsonl_path': '/scratch/s/sunyh/data/frozen_lake/reward_250.jsonl',
			'prompt_path': '/scratch/s/sunyh/data/frozen_lake/reward_prompt.txt',
			'num_total_samples': 1992
		},
        'frozen_lake_reward_500_trajs': {
			'data_dir': '/scratch/s/sunyh/data/frozen_lake/vlm_sft_images',
			'jsonl_path': '/scratch/s/sunyh/data/frozen_lake/reward_500.jsonl',
			'prompt_path': '/scratch/s/sunyh/data/frozen_lake/reward_prompt.txt',
			'num_total_samples': 3947
		},
        'mini_behavior_reward_100_trajs': {
            'data_dir': '/scratch/s/sunyh/data/mini_behavior/reward_images',
			'jsonl_path': '/scratch/s/sunyh/data/mini_behavior/reward_100.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/mini_behavior/reward_prompt.txt',
			'num_total_samples': 2908
        },
        'mini_behavior_reward_250_trajs': {
            'data_dir': '/scratch/s/sunyh/data/mini_behavior/reward_images',
			'jsonl_path': '/scratch/s/sunyh/data/mini_behavior/reward_250.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/mini_behavior/reward_prompt.txt',
			'num_total_samples': 5755
        },
        'mini_behavior_reward_500_trajs': {
            'data_dir': '/scratch/s/sunyh/data/mini_behavior/reward_images',
			'jsonl_path': '/scratch/s/sunyh/data/mini_behavior/reward_500.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/mini_behavior/reward_prompt.txt',
			'num_total_samples': 14390
        },
        'language_table_policy': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/policy.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/policy_prompt.txt',
			'num_total_samples': 18476
        },
        'language_table_reward': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/reward.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/reward_prompt.txt',
			'num_total_samples': 20476
        },
        'language_table_reward_100_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/reward_100.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/reward_prompt.txt',
			'num_total_samples': 1026
        },
        'language_table_reward_250_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/reward_250.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/reward_prompt.txt',
			'num_total_samples': 2565
        },
        'language_table_reward_500_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/reward_500.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/reward_prompt.txt',
			'num_total_samples': 5163
        },
        'language_table_reward_1000_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/reward_1000.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/reward_prompt.txt',
			'num_total_samples': 10273
        },
        'language_table_inverse_dynamics': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/inverse_dynamics.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/inverse_dynamics_prompt.txt',
			'num_total_samples': 18476
        },
        'language_table_inverse_dynamics_100_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/inverse_dynamics_100.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/inverse_dynamics_prompt.txt',
			'num_total_samples': 926
        },
        'language_table_inverse_dynamics_250_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/inverse_dynamics_250.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/inverse_dynamics_prompt.txt',
			'num_total_samples': 2315
        },
        'language_table_inverse_dynamics_500_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/inverse_dynamics_500.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/inverse_dynamics_prompt.txt',
			'num_total_samples': 4663
        },
        'language_table_inverse_dynamics_1000_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/inverse_dynamics_1000.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/inverse_dynamics_prompt.txt',
			'num_total_samples': 9273
        },
        'language_table_count_blocks': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/count_blocks.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/count_blocks_prompt.txt',
			'num_total_samples': 20476
        },
        'language_table_count_blocks_100_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/count_blocks_100.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/count_blocks_prompt.txt',
			'num_total_samples': 1026
        },
        'language_table_count_blocks_250_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/count_blocks_250.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/count_blocks_prompt.txt',
			'num_total_samples': 2565
        },
        'language_table_count_blocks_500_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/count_blocks_500.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/count_blocks_prompt.txt',
			'num_total_samples': 5163
        },
        'language_table_count_blocks_1000_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/count_blocks_1000.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/count_blocks_prompt.txt',
			'num_total_samples': 10273
        },
        'real_world_policy': {
            'data_dir': '/scratch/s/sunyh/data/real_world/images',
			'jsonl_path': '/scratch/s/sunyh/data/real_world/policy.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/real_world/policy_prompt.txt',
			'num_total_samples': 720
        },
        'real_world_reward': {
            'data_dir': '/scratch/s/sunyh/data/real_world/images',
			'jsonl_path': '/scratch/s/sunyh/data/real_world/reward.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/real_world/reward_prompt.txt',
			'num_total_samples': 890
        },
        'real_world_inverse_dynamics': {
            'data_dir': '/scratch/s/sunyh/data/real_world/images',
			'jsonl_path': '/scratch/s/sunyh/data/real_world/inverse_dynamics.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/real_world/inverse_dynamics_prompt.txt',
			'num_total_samples': 720
        },
        'frozen_lake_bagel_vlm_500_trajs': {
            'data_dir': '/scratch/s/sunyh/vlm_sft_data/frozen_lake/images',
			'jsonl_path': '/scratch/s/sunyh/vlm_sft_data/frozen_lake/bagel_vlm_data_500.jsonl',
            'prompt_path': '/scratch/s/sunyh/vlm_sft_data/frozen_lake/vlm_plan_prompt.txt',
			'num_total_samples': 3646
        },
        'frozen_lake_bagel_vlm_500_trajs_cot': {
            'data_dir': '/scratch/s/sunyh/vlm_sft_data/frozen_lake/images',
			'jsonl_path': '/scratch/s/sunyh/vlm_sft_data/frozen_lake/bagel_vlm_data_cot_500.jsonl',
            'prompt_path': '/scratch/s/sunyh/vlm_sft_data/frozen_lake/vlm_plan_cot_prompt.txt',
			'num_total_samples': 3646
        },
        'frozen_lake_bagel_vlm_1000_trajs_cot': {
            'data_dir': '/scratch/s/sunyh/vlm_sft_data/frozen_lake/images',
			'jsonl_path': '/scratch/s/sunyh/vlm_sft_data/frozen_lake/bagel_vlm_data_cot_1000.jsonl',
            'prompt_path': '/scratch/s/sunyh/vlm_sft_data/frozen_lake/vlm_plan_cot_prompt.txt',
			'num_total_samples': 7324
        },
        'frozen_lake_bagel_vlm_2000_trajs_cot': {
            'data_dir': '/scratch/s/sunyh/vlm_sft_data/frozen_lake/images',
			'jsonl_path': '/scratch/s/sunyh/vlm_sft_data/frozen_lake/bagel_vlm_data_cot_2000.jsonl',
            'prompt_path': '/scratch/s/sunyh/vlm_sft_data/frozen_lake/vlm_plan_cot_prompt.txt',
			'num_total_samples': 14614
        },
        'mini_behavior_bagel_vlm_500_trajs': {
            'data_dir': '/scratch/s/sunyh/vlm_sft_data/mini_behavior/images',
			'jsonl_path': '/scratch/s/sunyh/vlm_sft_data/mini_behavior/bagel_vlm_data_500.jsonl',
            'prompt_path': '/scratch/s/sunyh/vlm_sft_data/mini_behavior/vlm_plan_prompt.txt',
			'num_total_samples': 3854
        },
        'mini_behavior_bagel_vlm_500_trajs_cot': {
            'data_dir': '/scratch/s/sunyh/vlm_sft_data/mini_behavior/images',
			'jsonl_path': '/scratch/s/sunyh/vlm_sft_data/mini_behavior/bagel_vlm_data_cot_500.jsonl',
            'prompt_path': '/scratch/s/sunyh/vlm_sft_data/mini_behavior/vlm_plan_cot_prompt.txt',
			'num_total_samples': 3854
        },
        'mini_behavior_bagel_vlm_1000_trajs_cot': {
            'data_dir': '/scratch/s/sunyh/vlm_sft_data/mini_behavior/images',
			'jsonl_path': '/scratch/s/sunyh/vlm_sft_data/mini_behavior/bagel_vlm_data_cot_1000.jsonl',
            'prompt_path': '/scratch/s/sunyh/vlm_sft_data/mini_behavior/vlm_plan_cot_prompt.txt',
			'num_total_samples': 7812
        },
        'mini_behavior_bagel_vlm_2000_trajs_cot': {
            'data_dir': '/scratch/s/sunyh/vlm_sft_data/mini_behavior/images',
			'jsonl_path': '/scratch/s/sunyh/vlm_sft_data/mini_behavior/bagel_vlm_data_cot_2000.jsonl',
            'prompt_path': '/scratch/s/sunyh/vlm_sft_data/mini_behavior/vlm_plan_cot_prompt.txt',
			'num_total_samples': 15595
        },
        'language_table_bagel_vlm_500_trajs': {
            'data_dir': '/scratch/s/sunyh/vlm_sft_data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/vlm_sft_data/language_table_sim_v2/bagel_vlm_data_500.jsonl',
            'prompt_path': '/scratch/s/sunyh/vlm_sft_data/language_table_sim_v2/vlm_plan_prompt.txt',
			'num_total_samples': 5931
        },
        'language_table_bagel_vlm_500_trajs_cot': {
            'data_dir': '/scratch/s/sunyh/vlm_sft_data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/vlm_sft_data/language_table_sim_v2/bagel_vlm_data_cot_500.jsonl',
            'prompt_path': '/scratch/s/sunyh/vlm_sft_data/language_table_sim_v2/vlm_plan_cot_prompt.txt',
			'num_total_samples': 5931
        },
        'language_table_bagel_vlm_1000_trajs_cot': {
            'data_dir': '/scratch/s/sunyh/vlm_sft_data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/vlm_sft_data/language_table_sim_v2/bagel_vlm_data_cot_1000.jsonl',
            'prompt_path': '/scratch/s/sunyh/vlm_sft_data/language_table_sim_v2/vlm_plan_cot_prompt.txt',
			'num_total_samples': 12021
        },
        'language_table_bagel_vlm_2000_trajs_cot': {
            'data_dir': '/scratch/s/sunyh/vlm_sft_data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/vlm_sft_data/language_table_sim_v2/bagel_vlm_data_cot_2000.jsonl',
            'prompt_path': '/scratch/s/sunyh/vlm_sft_data/language_table_sim_v2/vlm_plan_cot_prompt.txt',
			'num_total_samples': 24104
        },
    },
    'dynamics_sft': {
        'frozen_lake_100_trajs': {
            'data_dir': '/scratch/s/sunyh/data/frozen_lake/dynamics_images',
			'jsonl_path': '/scratch/s/sunyh/data/frozen_lake/dynamics_100.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/frozen_lake/dynamics_prompt.txt',
			'num_total_samples': 707
        },
        'frozen_lake_250_trajs': {
            'data_dir': '/scratch/s/sunyh/data/frozen_lake/dynamics_images',
			'jsonl_path': '/scratch/s/sunyh/data/frozen_lake/dynamics_250.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/frozen_lake/dynamics_prompt.txt',
			'num_total_samples': 1742
        },
        'frozen_lake_500_trajs': {
            'data_dir': '/scratch/s/sunyh/data/frozen_lake/dynamics_images',
			'jsonl_path': '/scratch/s/sunyh/data/frozen_lake/dynamics_500.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/frozen_lake/dynamics_prompt.txt',
			'num_total_samples': 3438
        },
        'mini_behavior_100_trajs': {
            'data_dir': '/scratch/s/sunyh/data/mini_behavior/dynamics_images',
			'jsonl_path': '/scratch/s/sunyh/data/mini_behavior/dynamics_100.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/mini_behavior/dynamics_prompt.txt',
			'num_total_samples': 774
        },
        'mini_behavior_250_trajs': {
            'data_dir': '/scratch/s/sunyh/data/mini_behavior/dynamics_images',
			'jsonl_path': '/scratch/s/sunyh/data/mini_behavior/dynamics_250.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/mini_behavior/dynamics_prompt.txt',
			'num_total_samples': 1933
        },
        'mini_behavior_500_trajs': {
            'data_dir': '/scratch/s/sunyh/data/mini_behavior/dynamics_images',
			'jsonl_path': '/scratch/s/sunyh/data/mini_behavior/dynamics_500.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/mini_behavior/dynamics_prompt.txt',
			'num_total_samples': 3940
        },
        'language_table_block2block': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_block2block.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_prompt.txt',
			'num_total_samples': 11822
        },
        'language_table_block2block_100_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_block2block_100.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_prompt.txt',
			'num_total_samples': 152
        },
        'language_table_block2block_250_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_block2block_250.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_prompt.txt',
			'num_total_samples': 738
        },
        'language_table_block2block_500_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_block2block_500.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_prompt.txt',
			'num_total_samples': 1520
        },
        'language_table_block2block_1000_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_block2block_1000.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_prompt.txt',
			'num_total_samples': 2951
        },
        'language_table_block2pos': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_block2pos.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_prompt.txt',
			'num_total_samples': 12565
        },
        'language_table_block2pos_100_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_block2pos_100.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_prompt.txt',
			'num_total_samples': 310
        },
        'language_table_block2pos_250_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_block2pos_250.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_prompt.txt',
			'num_total_samples': 1577
        },
        'language_table_block2pos_500_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_block2pos_500.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_prompt.txt',
			'num_total_samples': 3143
        },
        'language_table_block2pos_1000_trajs': {
            'data_dir': '/scratch/s/sunyh/data/language_table_sim_v2/images',
			'jsonl_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_block2pos_1000.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/language_table_sim_v2/dynamics_prompt.txt',
			'num_total_samples': 6322
        },
        'real_world': {
            'data_dir': '/scratch/s/sunyh/data/real_world/images',
			'jsonl_path': '/scratch/s/sunyh/data/real_world/dynamics.jsonl',
            'prompt_path': '/scratch/s/sunyh/data/real_world/dynamics_prompt.txt',
			'num_total_samples': 720
        },
    }
}