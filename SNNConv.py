import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Micro Hei',
    'WenQuanYi Zen Hei',
    'Noto Sans CJK SC',
    'Noto Sans CJK JP',
    'Noto Serif CJK SC',
    'SimHei',
    'Microsoft YaHei',
    'Arial Unicode MS',
    'DejaVu Sans',
]
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 20,          # 全局默认字号
    'axes.titlesize': 20,     # 子图标题
    'axes.labelsize': 20,     # 坐标轴标题（x/y label）
    'xtick.labelsize': 18,    # x 轴刻度
    'ytick.labelsize': 18,    # y 轴刻度
    'legend.fontsize': 20,    # 图例
    'figure.titlesize': 20,   # 整张图的大标题（如果用到）
})
import h5py
from pathlib import Path
from typing import Optional, Union
from spikingjelly.activation_based import neuron, surrogate, functional, layer

# ==========================================
# 1. Defined Adaptive Threshold Neuron (ALIF Node)
# ==========================================
class ALIFNode(neuron.BaseNode):
    def __init__(self, v_threshold=1.0, tau=2.0, beta=1.0, detach_reset=True):
        super().__init__(v_threshold=v_threshold, v_reset=0., surrogate_function=surrogate.ATan(), detach_reset=detach_reset, step_mode='m')
        
        self.tau = tau
        self.decay = 1.0 - (1.0 / tau)
        self.beta = beta
        self.trace_decay = 0.9  
        
        self.register_memory('v_mem', 0.)
        self.register_memory('trace', 0.)
        
        # [新增] 用于记录重置前的电压（即冲过阈值那一瞬间的电压）
        self.v_mem_pre = None 

    def forward(self, x: torch.Tensor):
        if self.step_mode == 's':
            return self.single_step_forward(x)
        
        # step_mode='m' 逻辑保持不变...
        time_dim = 0 
        T = x.shape[time_dim]
        spikes = []
        for t in range(T):
            xt = x.select(time_dim, t) 
            s = self.single_step_forward(xt)
            spikes.append(s)
        return torch.stack(spikes, dim=time_dim)

    def single_step_forward(self, x: torch.Tensor):
        # 1. 积分
        self.v_mem = self.v_mem * self.decay + x
        
        # 2. 计算自适应阈值
        adaptive_thresh = self.v_threshold + self.beta * self.trace
        if not isinstance(adaptive_thresh, torch.Tensor):
            adaptive_thresh = torch.as_tensor(adaptive_thresh, device=x.device, dtype=x.dtype)
        
        # [新增] 在重置之前，先备份一下当前的电压！
        # 这就是那个“超过阈值”的高电压
        self.v_mem_pre = self.v_mem.clone()

        # 3. 发放
        spike = self.surrogate_function(self.v_mem - adaptive_thresh)
        
        # 4. 重置 (这里 v_mem 会变回 0)
        self.v_mem = self.v_mem * (1. - spike) 
        
        # 5. 更新 Trace
        self.trace = self.trace * self.trace_decay + spike
        
        return spike

# ==========================================
# 2. Convolutional ALIF Layer
# ==========================================
class SNNConvALIFLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, beta=1.5):
        super().__init__()
        
        # 1. Spatial Features: Conv2d + InstanceNorm
        # SeqToANNContainer allows standard nn.Modules to process [T, N, ...] inputs efficiently
        self.fwd = layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True) 
        )
        
        # 2. Temporal Spiking: ALIF
        self.act = ALIFNode(v_threshold=1.0, beta=beta, tau=2.0)

    def forward(self, x):
        # x shape: [T, N, C, H, W]
        x = self.fwd(x) 
        x = self.act(x)
        return x

# ==========================================
# 3. Modified Network Architecture (Net)
# ==========================================
# ==========================================
# 2. 修改后的 Net (记录 v_pre)
# ==========================================
class Net(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        C, H, W = input_shape
        print(f"Network Initialized. Input Shape: {input_shape}")
        
        self.conv_layer = SNNConvALIFLayer(
            in_channels=C, 
            out_channels=16, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            beta=2.0 
        )
        
        self.readout = layer.SeqToANNContainer(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),            
            nn.Linear(16, 10)        
        )

    def forward(self, x):
        T = x.shape[0]
        
        monitor = {
            'spike': [],
            'v_mem': [],    # 重置后 (0 或 负数)
            'v_pre': [],    # 重置前 (冲锋值)
            'thresh': []    # 判定时的阈值
        }
        
        # 重置网络状态
        functional.reset_net(self)
        
        # 强制单步模式，以便手动循环
        self.conv_layer.act.step_mode = 's'
        
        for t in range(T):
            xt = x[t]
            # 1. 卷积提取特征 (增加 batch 维度)
            feat = self.conv_layer.fwd(xt.unsqueeze(0))[0]
            
            # ==========================================
            # [关键修改] 先计算并记录“当前生效”的阈值
            # ==========================================
            # 此时 trace 还是上一时刻的值，正好对应当前这一步的判定标准
            # 确保 trace 是 tensor (SpikingJelly 内部通常是 tensor)
            trace_val = self.conv_layer.act.trace
            if not isinstance(trace_val, torch.Tensor):
                trace_val = torch.tensor(trace_val, device=feat.device)
                
            effective_thresh = self.conv_layer.act.v_threshold + \
                               self.conv_layer.act.beta * trace_val
            
            if not isinstance(effective_thresh, torch.Tensor):
                 effective_thresh = torch.tensor(effective_thresh, device=feat.device)

            # --- [新增的关键修复] ---
            # 如果阈值还是标量（比如 t=0 时），强制把它扩展成和 feat 一样的形状
            if effective_thresh.ndim == 0:
                effective_thresh = effective_thresh.expand_as(feat)
            # ----------------------

            monitor['thresh'].append(effective_thresh.detach().cpu())
            # ==========================================

            # 2. 执行神经元动力学 (这会更新 trace)
            s = self.conv_layer.act(feat)
            
            # 3. 记录其他数据
            monitor['spike'].append(s.detach().cpu())
            
            # 记录重置前的冲锋电压 (需配合修改后的 ALIFNode)
            monitor['v_pre'].append(self.conv_layer.act.v_mem_pre.detach().cpu())
            
            # 记录重置后的电压 (保持数据完整性)
            monitor['v_mem'].append(self.conv_layer.act.v_mem.detach().cpu())
            
        # 堆叠数据
        for k in monitor:
            monitor[k] = torch.stack(monitor[k])
            
        return monitor

# ==========================================
# 4. Data Loading (Unchanged)
# ==========================================
def load_h5_voxel_grid(file_path, key='data'):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with h5py.File(file_path, 'r') as f:
        if key not in f.keys():
            print(f"Warning: Key '{key}' not found. Available: {list(f.keys())}")
            key = list(f.keys())[0]
        data = f[key][:] 

    tensor_data = torch.from_numpy(data).float()
    if tensor_data.ndim == 4: 
        tensor_data = tensor_data.unsqueeze(1)
    elif tensor_data.ndim == 3:
        tensor_data = tensor_data.unsqueeze(1).unsqueeze(1)
    
    print(f"Loaded {file_path.name}: Shape {tensor_data.shape}, Max Val: {tensor_data.max()}")
    return tensor_data

# ==========================================
# 5. Run Experiment
# ==========================================
def run_comparison(normal_path, flash_path, save_path: Optional[Union[str, Path]] = None):
    # 1. Load Data
    print(f"Loading Normal Data from: {normal_path}")
    data_normal = load_h5_voxel_grid(normal_path)
    print(f"Loading Flash Data from: {flash_path}")
    data_flash = load_h5_voxel_grid(flash_path)

    assert data_normal.shape == data_flash.shape, "Error: Shapes must match!"
    T = data_normal.shape[0]
    C, H, W = data_normal.shape[2], data_normal.shape[3], data_normal.shape[4]

    denom = float(C * H * W)

    normal_event_density = (data_normal.abs().sum(dim=(1, 2, 3, 4)) / denom).cpu().numpy()
    flash_event_density = (data_flash.abs().sum(dim=(1, 2, 3, 4)) / denom).cpu().numpy()

    normal_nonzero_ratio = (data_normal != 0).float().mean(dim=(1, 2, 3, 4)).cpu().numpy()
    flash_nonzero_ratio = (data_flash != 0).float().mean(dim=(1, 2, 3, 4)).cpu().numpy()

    print("Per-timestep event density: abs-sum / (C*H*W)")
    for t in range(T):
        print(
            f"t={t:03d}  normal={normal_event_density[t]:.6g}  flash={flash_event_density[t]:.6g}"
            f"  nonzero(normal)={normal_nonzero_ratio[t]:.6g}  nonzero(flash)={flash_nonzero_ratio[t]:.6g}"
        )

    time_axis = np.arange(T)

    fig_stats, ax_stats = plt.subplots(1, 1, figsize=(10, 5))

    normal_density_plot = np.clip(normal_event_density, 1e-12, None)
    flash_density_plot = np.clip(flash_event_density, 1e-12, None)

    ax_stats.plot(time_axis, normal_density_plot, 'b-o', label='正常条件')
    ax_stats.plot(time_axis, flash_density_plot, 'r-o', label='突发异常')
    ax_stats.set_yscale('log')
    ax_stats.set_ylabel('事件密度')
    ax_stats.set_xlabel('时间步')

    for t_mark in (3, 6):
        ax_stats.axvline(x=t_mark, linestyle='--', color='k', linewidth=2, alpha=0.9)

    ax_stats.grid(True, alpha=0.3)
    ax_stats.legend()

    fig_stats.tight_layout()

    # 创建输出文件夹
    output_dir = Path(__file__).parent / 'snn_analysis_results'
    output_dir.mkdir(exist_ok=True)
    
    stats_save_path = output_dir / 'event_density_stats.png'

    fig_stats.savefig(stats_save_path, dpi=200, bbox_inches='tight')
    plt.close(fig_stats)
    print(f"Saved stats figure to: {stats_save_path.resolve()}")

    # 2. Initialize Net with Conv Layer
    net = Net(input_shape=(C, H, W))
    
    # --- Normal ---
    print("Processing Normal Data...")
    rec_normal = net(data_normal)
    
    # --- Flash ---
    print("Processing Flash Data...")
    rec_flash = net(data_flash)
    
    # ================= Plotting (Individual Plots) =================
    time_axis = np.arange(T)
    output_dir = Path(__file__).parent / 'snn_analysis_results'
    output_dir.mkdir(exist_ok=True)

    # 1. Input Voxel Sum Comparison
    fig1, axs1 = plt.subplots(1, 2, figsize=(12, 5))
    axs1[0].plot(time_axis, data_normal.sum(dim=(1,2,3,4)).numpy(), 'b-o')
    axs1[0].set_title('正常条件')
    axs1[0].set_xlabel('时间步')
    axs1[0].set_ylabel('事件体素数量')
    axs1[0].grid(True, alpha=0.3)

    axs1[1].plot(time_axis, data_flash.sum(dim=(1,2,3,4)).numpy(), 'r-o')
    axs1[1].set_title('突发异常')
    axs1[1].set_xlabel('时间步')
    axs1[1].set_ylabel('事件体素数量')

    for t_mark in (3, 6):
        axs1[1].axvline(x=t_mark, linestyle='--', color='k', linewidth=2, alpha=0.9)

    axs1[1].grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(output_dir / '1_voxel_sum_comparison.png', dpi=200)
    plt.close(fig1)


    # 3. Mean Membrane Potential Comparison
    fig3, axs3 = plt.subplots(1, 2, figsize=(12, 5))
    v_mean_norm = rec_normal['v_mem'].mean(dim=(1,2,3,4)).numpy()
    v_mean_flash = rec_flash['v_mem'].mean(dim=(1,2,3,4)).numpy()
    
    axs3[0].plot(time_axis, v_mean_norm, 'b-o')
    axs3[0].set_title('正常条件')
    axs3[0].set_xlabel('时间步')
    axs3[0].set_ylabel('平均膜电位')
    axs3[0].grid(True)

    axs3[1].plot(time_axis, v_mean_flash, 'r-o')
    axs3[1].set_title('突发异常')
    axs3[1].set_xlabel('时间步')
    axs3[1].set_ylabel('平均膜电位')

    for t_mark in (3, 6):
        axs3[1].axvline(x=t_mark, linestyle='--', color='k', linewidth=2, alpha=0.9)

    axs3[1].grid(True)
    fig3.tight_layout()
    fig3.savefig(output_dir / '3_vmem_comparison.png', dpi=200)
    plt.close(fig3)

    # 4. Adaptive Threshold Comparison
    fig4, axs4 = plt.subplots(1, 2, figsize=(12, 5))
    th_mean_norm = rec_normal['thresh'].mean(dim=(1,2,3,4)).numpy()
    th_mean_flash = rec_flash['thresh'].mean(dim=(1,2,3,4)).numpy()
    
    axs4[0].plot(time_axis, th_mean_norm, 'b--', label='自适应阈值')
    axs4[0].set_title('正常条件')
    axs4[0].set_xlabel('时间步')
    axs4[0].set_ylabel('阈值水平')
    axs4[0].legend()
    axs4[0].grid(True, alpha=0.3)

    axs4[1].plot(time_axis, th_mean_flash, 'r--', linewidth=2, label='自适应阈值')
    axs4[1].set_title('突发异常')
    axs4[1].set_xlabel('时间步')
    axs4[1].set_ylabel('阈值水平')
    axs4[1].legend()

    axs4[1].axvline(x=3, linestyle='--', color='k', linewidth=2, alpha=0.9)


    axs4[1].grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(output_dir / '4_threshold_comparison.png', dpi=200)
    plt.close(fig4)

    print(f"Saved individual comparison plots to: {output_dir}")


if __name__ == "__main__":
    NORMAL_H5_PATH = "/home/zy/data/zy/zhaoyue/Fastchange/Datasets/Fan_sorted_events/my_voxel_results_768/events_voxel.h5" 
    FLASH_H5_PATH = "/home/zy/data/zy/zhaoyue/Fastchange/Datasets/Fan_events/my_voxel_results_768/events_voxel.h5"  
    
    if not Path(NORMAL_H5_PATH).exists():
        print("File not found, skipping execution.")
    else:
        run_comparison(NORMAL_H5_PATH, FLASH_H5_PATH)