1. 给原始图像数据添加时间戳
python make_timestamps.py

2. 添加光照
python image_light.py


3. 将整个序列生成事件数据
python fast_change.py single_sequence \
  --sequence_dir /home/zy/zhaoyue/Fastchange/Datasets/Fan_sorted_global \
  --output_events_dir /home/zy/zhaoyue/Fastchange/Datasets/Fan_events \
  --frame_rate 4000

python fast_change.py single_sequence \
  --sequence_dir /home/zy/zhaoyue/Fastchange/Datasets/Fan_sorted \
  --output_events_dir /home/zy/zhaoyue/Fastchange/Datasets/Fan_sorted_events \
  --frame_rate 4000


4. 体素网格/可视化事件流数据
python "/home/zy/zhaoyue/Fastchange/voxel gird_visualize.py" \
  --input "/home/zy/zhaoyue/Fastchange/Datasets/Fan_sorted_events/Fan_sorted/events.h5" \
  --output "/home/zy/zhaoyue/Fastchange/Datasets/Fan_sorted_events/my_voxel_results" \
  --bins 100 --fps 10 --no-images

python "/home/zy/zhaoyue/Fastchange/voxel gird_visualize.py" \
  --input "/home/zy/zhaoyue/Fastchange/Datasets/Fan_sorted_events/Fan_sorted/events.h5" \
  --output "/home/zy/zhaoyue/Fastchange/Datasets/Fan_sorted_events/my_voxel_results_768" \
  --bins 10 \
  --save-h5 \
  --width 768 --height 768 \
  --no-video --no-images



python /home/zy/zhaoyue/Fastchange/inspect_h5.py --h5 /home/zy/zhaoyue/Fastchange/Datasets/Fan_events/my_voxel_results_768/events_voxel.h5


2. 训练SNN模型
python SNN.py
python SNNConv.py
python SNNConv-STDP.py



