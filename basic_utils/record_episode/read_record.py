import re
import os


def read_record(continue_path, flag_once=False):
    """
    Read navigation metrics from evaluation record file
    
    This function parses evaluation records to extract navigation performance
    metrics including success rate, SPL (Success weighted by Path Length), 
    and distance measurements for continuous evaluation.
    
    Args:
        continue_path (str): Path to the evaluation record file
        flag_once (bool): If True, return empty metrics (single run mode)
        
    Returns:
        tuple: (num_total, num_success, spl_all, soft_spl_all, 
                distance_to_goal_all, distance_to_goal_reward_all, last_time)
                
    Metrics returned:
        - num_total: Total number of episodes completed
        - num_success: Number of successful episodes  
        - spl_all: Cumulative Success weighted by Path Length
        - soft_spl_all: Cumulative Soft SPL metric
        - distance_to_goal_all: Cumulative distance to goal
        - distance_to_goal_reward_all: Cumulative distance-based rewards
        - last_time: Time spent on last episode
    """
    # Initialize variables to store the metrics
    num_total = 0
    num_success = 0
    spl_all = 0.0
    soft_spl_all = 0.0
    distance_to_goal_all = 0.0
    distance_to_goal_reward_all = 0.0
    last_time = 0.0

    # Return directly for single run
    if flag_once:
        return (
            num_total,
            num_success,
            spl_all,
            soft_spl_all,
            distance_to_goal_all,
            distance_to_goal_reward_all,
            last_time,
        )
    # Confirm file exists
    if os.path.exists(continue_path):
        with open(continue_path, "r") as file:
            content = file.read()

        # Use regex to extract the topmost data block
        records = re.split(r"Scene ID: ", content)
        if len(records) > 1:
            latest_record = records[1]  # Take the first record

            # Extract total metrics
            num_success = int(
                re.search(r"Total Success\s+\|\s+(\d+)", latest_record).group(1)
            )
            spl_all = float(
                re.search(r"Total SPL\s+\|\s+([\d\.]+)", latest_record).group(1)
            )
            soft_spl_all = float(
                re.search(r"Total Soft SPL\s+\|\s+([\d\.]+)", latest_record).group(1)
            )
            distance_to_goal_all = float(
                re.search(
                    r"Total Distance to Goal\s+\|\s+([\d\.]+)", latest_record
                ).group(1)
            )

            # Extract task number and time spent
            num_total = int(
                re.search(r"No\.(\d+) task is finished", latest_record).group(1)
            )
            last_time = float(
                re.search(
                    r"(\d+\.\d+) seconds spend in this task", latest_record
                ).group(1)
            )

        print("Successfully read the previous set of metrics")

    return (
        num_total,
        num_success,
        spl_all,
        soft_spl_all,
        distance_to_goal_all,
        distance_to_goal_reward_all,
        last_time,
    )
