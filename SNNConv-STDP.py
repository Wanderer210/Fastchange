import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg') # 后台绘图，不弹窗
import matplotlib.pyplot as plt
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Union
from spikingjelly.activation_based import neuron, surrogate, functional, layer

# ==========================================
# 1. Defined Adaptive Threshold Neuron (ALIF Node)
#    [改进] 增加了 v_mem_pre 记录，用于可视化超过阈值的那一瞬间
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
        
        # [新增] 用于记录重置前的电压
        self.v_mem_pre = None 

    def forward(self, x: torch.Tensor):
        if self.step_mode == 's':
            return self.single_step_forward(x)
        
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
        
        # 2. 计算自适应阈值 (仅用于判定，不在此处更新 trace)
        adaptive_thresh = self.v_threshold + self.beta * self.trace
        if not isinstance(adaptive_thresh, torch.Tensor):
            adaptive_thresh = torch.as_tensor(adaptive_thresh, device=x.device, dtype=x.dtype)
        
        # [关键] 备份重置前的电压
        self.v_mem_pre = self.v_mem.clone()

        # 3. 发放脉冲
        spike = self.surrogate_function(self.v_mem - adaptive_thresh)
        
        # 4. 重置电压
        self.v_mem = self.v_mem * (1. - spike) 
        
        # 5. 更新 Trace (为下一步做准备)
        self.trace = self.trace * self.trace_decay + spike
        
        return spike

# ==========================================
# 2. Convolutional ALIF Layer
# ==========================================
class SNNConvALIFLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, beta=1.5):
        super().__init__()
        self.fwd = layer.SeqToANNContainer(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True) 
        )
        self.act = ALIFNode(v_threshold=1.0, beta=beta, tau=2.0)

    def forward(self, x):
        x = self.fwd(x) 
        x = self.act(x)
        return x

# ==========================================
# 3. Modified Network Architecture (Net)
#    [改进] 修复了阈值记录时序，解决了 torch.stack 报错
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

    def forward(self, x):
        T = x.shape[0]
        
        monitor = {
            'spike': [],
            'v_mem': [],    # 重置后 (0 或 负数)
            'v_pre': [],    # 重置前 (冲锋值)
            'thresh': []    # 判定时的阈值
        }
        
        functional.reset_net(self)
        self.conv_layer.act.step_mode = 's'
        
        for t in range(T):
            xt = x[t]
            # 1. 卷积提取特征
            feat = self.conv_layer.fwd(xt.unsqueeze(0))[0]
            
            # ==========================================
            # [关键修复] 先计算并记录“当前生效”的阈值
            # ==========================================
            trace_val = self.conv_layer.act.trace
            if not isinstance(trace_val, torch.Tensor):
                trace_val = torch.tensor(trace_val, device=feat.device)
                
            effective_thresh = self.conv_layer.act.v_threshold + \
                               self.conv_layer.act.beta * trace_val
            
            if not isinstance(effective_thresh, torch.Tensor):
                 effective_thresh = torch.tensor(effective_thresh, device=feat.device)

            # [修复 RuntimeError] 如果阈值是标量(t=0)，强制广播形状
            if effective_thresh.ndim == 0:
                effective_thresh = effective_thresh.expand_as(feat)

            monitor['thresh'].append(effective_thresh.detach().cpu())
            # ==========================================

            # 2. 执行神经元动力学
            s = self.conv_layer.act(feat)
            
            # 3. 记录其他数据
            monitor['spike'].append(s.detach().cpu())
            monitor['v_pre'].append(self.conv_layer.act.v_mem_pre.detach().cpu())
            monitor['v_mem'].append(self.conv_layer.act.v_mem.detach().cpu())
            
        for k in monitor:
            monitor[k] = torch.stack(monitor[k])
            
        return monitor

# ==========================================
# 4. Data Loading
# ==========================================
def load_h5_voxel_grid(file_path, key='data'):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with h5py.File(file_path, 'r') as f:
        if key not in f.keys():
            key = list(f.keys())[0]
        data = f[key][:] 
    tensor_data = torch.from_numpy(data).float()
    if tensor_data.ndim == 4: 
        tensor_data = tensor_data.unsqueeze(1)
    elif tensor_data.ndim == 3:
        tensor_data = tensor_data.unsqueeze(1).unsqueeze(1)
    print(f"Loaded {file_path.name}: Shape {tensor_data.shape}")
    return tensor_data

# ==========================================
# 5. Advanced Visualization Functions
# ==========================================
def plot_advanced_analysis(rec_normal, rec_flash, T, save_path_prefix):
    """ 绘制空间热力图和通道响应图 """
    
    # --- A. 空间热力图 ---
    spatial_map_normal = rec_normal['spike'].sum(dim=(0, 2)).squeeze().numpy()
    spatial_map_flash = rec_flash['spike'].sum(dim=(0, 2)).squeeze().numpy()
    global_max = max(spatial_map_normal.max(), spatial_map_flash.max())
    if global_max == 0: global_max = 1

    fig_spatial, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(spatial_map_normal, cmap='hot', vmin=0, vmax=global_max)
    axs[0].set_title("Normal: Accumulated Spikes")
    axs[0].axis('off')
    im2 = axs[1].imshow(spatial_map_flash, cmap='hot', vmin=0, vmax=global_max)
    axs[1].set_title("Flash: Accumulated Spikes")
    axs[1].axis('off')
    plt.colorbar(im2, ax=axs, orientation='horizontal', fraction=0.05, pad=0.05, label='Spike Count')
    plt.suptitle("Spatial Distribution")
    
    save_path_spatial = str(save_path_prefix).replace('.png', '_spatial.png')
    plt.savefig(save_path_spatial, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_spatial}")

    # --- B. 通道级动态 ---
    ch_activity_normal = rec_normal['spike'].mean(dim=(3, 4)).squeeze(1).numpy()
    ch_activity_flash = rec_flash['spike'].mean(dim=(3, 4)).squeeze(1).numpy()
    time_axis = np.arange(T)
    num_channels = ch_activity_normal.shape[1]
    
    fig_ch, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    colors = plt.cm.jet(np.linspace(0, 1, num_channels))
    
    for c in range(num_channels):
        axs[0].plot(time_axis, ch_activity_normal[:, c], color=colors[c], alpha=0.7)
    axs[0].set_title(f"Normal Stream: Activity per Channel")
    axs[0].grid(True, alpha=0.3)

    for c in range(num_channels):
        axs[1].plot(time_axis, ch_activity_flash[:, c], color=colors[c], alpha=0.7)
    axs[1].set_title(f"Flash Stream: Activity per Channel")
    axs[1].grid(True, alpha=0.3)
    
    save_path_channel = str(save_path_prefix).replace('.png', '_channels.png')
    plt.savefig(save_path_channel, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_channel}")

def plot_single_neuron_dynamics(rec_flash, save_path_prefix):
    """ 显微镜模式：绘制单神经元 v_pre vs threshold """
    # 找到发放最多的神经元
    total_spikes_map = rec_flash['spike'].sum(dim=0).squeeze(0) 
    flat_idx = torch.argmax(total_spikes_map)
    C, H, W = total_spikes_map.shape
    idx_c = int(flat_idx // (H * W))
    rem = int(flat_idx % (H * W))
    idx_h = int(rem // W)
    idx_w = int(rem % W)
    
    print(f"Microscope Target: Channel={idx_c}, H={idx_h}, W={idx_w}")
    
    trace_v_pre = rec_flash['v_pre'][:, 0, idx_c, idx_h, idx_w].numpy()
    trace_thresh = rec_flash['thresh'][:, 0, idx_c, idx_h, idx_w].numpy()
    trace_spike = rec_flash['spike'][:, 0, idx_c, idx_h, idx_w].numpy()
    
    T = len(trace_v_pre)
    time_axis = np.arange(T)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, trace_thresh, 'k--', linewidth=2, label='Adaptive Threshold', alpha=0.8)
    plt.plot(time_axis, trace_v_pre, 'b-', linewidth=2, label='Membrane Potential (Pre-reset)')
    
    spike_times = time_axis[trace_spike > 0]
    spike_vals = trace_v_pre[trace_spike > 0]
    
    if len(spike_times) > 0:
        plt.scatter(spike_times, spike_vals, color='red', s=200, marker='*', zorder=10, label='Spike!')
        for t, v in zip(spike_times, spike_vals):
            plt.vlines(t, 0, v, colors='red', linestyles='dotted', alpha=0.5)

    plt.title(f"Microscope View: Single Neuron Dynamics (Ch={idx_c}, Pos={idx_h},{idx_w})", fontsize=14)
    plt.xlabel("Time Step")
    plt.ylabel("Voltage")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = str(save_path_prefix).replace('.png', '_microscope.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ==========================================
# 6. STDP Analysis
# ==========================================
class STDPMonitor:
    def __init__(self, height=32, width=32, tau_pre=2.0, tau_post=2.0):
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        
        # 内部状态 Trace
        self.trace_pre = None
        self.trace_post = None
        
        # [核心修改] 累积权重变化图 (H, W)
        # 我们将不同通道的变化求和，只看空间位置上的“学习热度”
        self.accumulated_delta_w = torch.zeros((height, width))

    def step(self, spike_pre, spike_post):
        """
        spike_pre: [Batch, In_Channels, H, W]
        spike_post: [Batch, Out_Channels, H, W]
        """
        # 1. 初始化
        if self.trace_pre is None:
            self.trace_pre = torch.zeros_like(spike_pre)
        if self.trace_post is None:
            self.trace_post = torch.zeros_like(spike_post)

        # 2. 更新 Trace (标准 STDP 衰减)
        self.trace_pre = self.trace_pre * (1 - 1/self.tau_pre) + spike_pre
        self.trace_post = self.trace_post * (1 - 1/self.tau_post) + spike_post
        
        # 3. 计算 ΔW (空间映射版)
        # 为了可视化 "哪个像素位置的突触正在发生改变"，我们在通道维度(dim=1)上求和
        # 这样我们将多通道的复杂交互简化为一张 HxW 的图
        
        # LTP (Pre Trace + Post Spike)
        # 我们假设同一个空间位置的输入和输出存在突触连接
        # sum(dim=1) 将 [1, C, H, W] -> [1, H, W]
        pre_trace_map = self.trace_pre.sum(dim=1).squeeze(0)
        post_spike_map = spike_post.sum(dim=1).squeeze(0)
        ltp_map = post_spike_map * pre_trace_map # 对应位置相乘

        # LTD (Post Trace + Pre Spike)
        post_trace_map = self.trace_post.sum(dim=1).squeeze(0)
        pre_spike_map = spike_pre.sum(dim=1).squeeze(0)
        ltd_map = pre_spike_map * post_trace_map

        # 4. 累积变化
        # Net Change = LTP (增强) - LTD (抑制)
        current_step_delta = ltp_map - ltd_map
        self.accumulated_delta_w += current_step_delta.detach().cpu()

def plot_weight_change_heatmap(monitor_normal, monitor_flash, save_path_prefix):
    """
    绘制累积的权重变化空间分布图
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    map_norm = monitor_normal.accumulated_delta_w.numpy()
    map_flash = monitor_flash.accumulated_delta_w.numpy()

    # 找出两个图中绝对值最大的数，用于统一色标范围，保证 0 在中间
    max_val = max(abs(map_norm.max()), abs(map_norm.min()), 
                  abs(map_flash.max()), abs(map_flash.min()))
    if max_val == 0: max_val = 1e-5
    
    # 定义绘图规范
    norm = colors.TwoSlopeNorm(vmin=-max_val, vcenter=0., vmax=max_val)
    cmap = 'seismic' # 红-白-蓝 配色 (红正，蓝负)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Normal Plot
    im1 = axs[0].imshow(map_norm, cmap=cmap, norm=norm)
    axs[0].set_title("Normal Stream: Accumulated $\Delta W$")
    axs[0].axis('off')
    
    # 2. Flash Plot
    im2 = axs[1].imshow(map_flash, cmap=cmap, norm=norm)
    axs[1].set_title("Flash Stream: Accumulated $\Delta W$")
    axs[1].axis('off')
    
    # Colorbar
    cbar = fig.colorbar(im2, ax=axs, orientation='vertical', fraction=0.05, pad=0.05)
    cbar.set_label('Weight Change (Red=LTP, Blue=LTD)')
    
    plt.suptitle("Spatial Map of Synaptic Learning Pressure", fontsize=14)
    
    save_path = str(save_path_prefix).replace('.png', '_weight_map.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved weight heatmap to: {save_path}")

def run_stdp_analysis(data_normal, data_flash, net, save_path):
    print("Running STDP Analysis...")
    
    # 获取空间尺寸
    _, _, _, H, W = data_normal.shape
    
    # 初始化改进后的监视器
    monitor_norm = STDPMonitor(height=H, width=W)
    monitor_flash = STDPMonitor(height=H, width=W)
    
    T = data_normal.shape[0]
    
    # 1. Normal Stream
    functional.reset_net(net)
    net.conv_layer.act.step_mode = 's'
    for t in range(T):
        # data_*[t] shape: [1, C, H, W]
        xt = data_normal[t]
        # 参照 Net.forward：用 T=1 的输入喂给 SeqToANNContainer，再取出该时间步的输出
        feat = net.conv_layer.fwd(xt.unsqueeze(0))[0]  # [1, 16, H, W]
        spike_post = net.conv_layer.act(feat)          # [1, 16, H, W]
        monitor_norm.step(spike_pre=xt, spike_post=spike_post)

    # 2. Flash Stream
    functional.reset_net(net)
    for t in range(T):
        xt = data_flash[t]
        feat = net.conv_layer.fwd(xt.unsqueeze(0))[0]
        spike_post = net.conv_layer.act(feat)
        monitor_flash.step(spike_pre=xt, spike_post=spike_post)
        
    # --- 绘图 1: 之前的时序曲线图 (保持不变，用于看整体趋势) ---
    # (这里略去之前的曲线绘图代码，或者保留它，看你需求)
    
    # --- 绘图 2: 新增的空间热力图 ---
    plot_weight_change_heatmap(monitor_norm, monitor_flash, save_path)

# ==========================================
# 7. Main Execution Loop
# ==========================================
def run_comparison(normal_path, flash_path, save_path=None):
    if save_path is None:
        save_path = Path(__file__).with_name('snn_analysis.png')
    
    # 1. Load Data
    data_normal = load_h5_voxel_grid(normal_path)
    data_flash = load_h5_voxel_grid(flash_path)
    T, _, C, H, W = data_normal.shape
    
    # 2. Basic Stats Plot
    fig_stats, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    denom = float(C * H * W)
    norm_density = (data_normal.abs().sum(dim=(1, 2, 3, 4)) / denom).cpu().numpy()
    flash_density = (data_flash.abs().sum(dim=(1, 2, 3, 4)) / denom).cpu().numpy()
    
    axs[0].plot(norm_density, 'g-o', label='Normal')
    axs[0].plot(flash_density, 'r-o', label='Flash')
    axs[0].set_yscale('log')
    axs[0].set_title('Input Event Density (Log Scale)')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot((data_normal != 0).float().mean(dim=(1,2,3,4)).cpu(), 'g-o', label='Nonzero Normal')
    axs[1].plot((data_flash != 0).float().mean(dim=(1,2,3,4)).cpu(), 'r-o', label='Nonzero Flash')
    axs[1].set_title('Sparsity (Non-zero Ratio)')
    axs[1].legend()
    axs[1].grid(True)
    
    stats_path = str(save_path).replace('.png', '_inputs.png')
    fig_stats.savefig(stats_path)
    plt.close()
    
    # 3. Run Network
    net = Net(input_shape=(C, H, W))
    print("Processing Normal Data...")
    rec_normal = net(data_normal)
    print("Processing Flash Data...")
    rec_flash = net(data_flash)
    
    # 4. Generate All Plots
    print("Generating plots...")
    plot_advanced_analysis(rec_normal, rec_flash, T, save_path)
    plot_single_neuron_dynamics(rec_flash, save_path)
    run_stdp_analysis(data_normal, data_flash, net, save_path)
    
    print("All visualizations completed successfully.")

if __name__ == "__main__":
    # 请修改这里的路径
    NORMAL_PATH = "/home/zy/zhaoyue/Fastchange/Datasets/Fan_sorted_events/my_voxel_results_768/events_voxel.h5"
    FLASH_PATH = "/home/zy/zhaoyue/Fastchange/Datasets/Fan_events/my_voxel_results_768/events_voxel.h5"
    
    if Path(NORMAL_PATH).exists():
        run_comparison(NORMAL_PATH, FLASH_PATH)
    else:
        print("Path not found. Please set NORMAL_PATH and FLASH_PATH.")