# MultipleMasksWithConfidence.msg


# 包含多个掩码，每个掩码是一个布尔数组
sensor_msgs/PointCloud2[] point_clouds

# 对应的多个置信度得分
float32[] confidence_scores

# label索引列表
int32[] label_indices