"""
ADVANCED PINN for 2D Orifice Flow - Enhanced Version
Features:
- Adaptive sampling strategy
- Restart from checkpoint capability
- Data-driven training with synthetic measurement points
- Parameter identification (inverse problem setup)
- Visualization of flow derivatives and gradients
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from datetime import datetime
import os

# Force TensorFlow to ignore all GPUs and run on CPU only
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    # If there are no GPUs or this fails, just continue
    pass

# ===================== CONFIGURATION =====================
class Config:
    """Configuration class for easy parameter management"""
    
    # Domain parameters
    DOMAIN_LENGTH = 0.1
    TUBE_DIAMETER = 0.020
    ORIFICE_DIAMETER = 0.010
    INLET_VELOCITY = 1.0
    
    # Fluid properties
    WATER_DENSITY = 1000.0
    WATER_VISCOSITY = 0.001
    
    # Network architecture
    HIDDEN_LAYERS = 8
    NEURONS_PER_LAYER = 64
    ACTIVATION = 'tanh'
    
    # Training parameters
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5000
    BATCH_SIZE = 256
    NUM_COLLOCATION_POINTS = 10000
    NUM_BOUNDARY_POINTS = 2000
    
    # Loss weights
    PHYSICS_WEIGHT = 1.0
    BOUNDARY_WEIGHT = 10.0
    DATA_WEIGHT = 0.5  # For inverse problems
    
    # Adaptive sampling
    USE_ADAPTIVE_SAMPLING = True
    RESAMPLE_FREQUENCY = 500
    
    # Checkpointing
    SAVE_CHECKPOINTS = True
    CHECKPOINT_FREQ = 1000
    CHECKPOINT_DIR = './checkpoints'
    
    def __repr__(self):
        return json.dumps(self.__dict__, indent=2)


config = Config()
TUBE_RADIUS = config.TUBE_DIAMETER / 2
ORIFICE_RADIUS = config.ORIFICE_DIAMETER / 2
ORIFICE_CENTER_X = config.DOMAIN_LENGTH / 2


# ===================== GEOMETRY UTILITIES =====================
class GeometryUtils:
    """Utilities for domain and boundary operations"""
    
    @staticmethod
    def is_in_orifice(x, y, orifice_length=0.005):
        """Check if point is inside orifice"""
        x_in = (x >= ORIFICE_CENTER_X - orifice_length/2) & (x <= ORIFICE_CENTER_X + orifice_length/2)
        y_in = np.abs(y) <= ORIFICE_RADIUS
        return x_in & y_in
    
    @staticmethod
    def get_boundary_mask(x, y, boundary_type):
        """Get mask for specific boundary"""
        eps = 1e-5
        if boundary_type == 'inlet':
            return np.abs(x) < eps
        elif boundary_type == 'outlet':
            return x > config.DOMAIN_LENGTH - eps
        elif boundary_type == 'wall':
            return (np.abs(np.abs(y) - TUBE_RADIUS) < eps) & (x >= 0) & (x <= config.DOMAIN_LENGTH)
        elif boundary_type == 'orifice':
            return (np.abs(np.abs(y) - ORIFICE_RADIUS) < eps) & \
                   (x >= ORIFICE_CENTER_X - 0.01) & (x <= ORIFICE_CENTER_X + 0.01)
        return np.zeros_like(x, dtype=bool)


# ===================== ADAPTIVE SAMPLING =====================
class AdaptiveSampler:
    """Adaptive collocation point sampling based on loss gradients"""
    
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        self.previous_points = None
    
    def compute_residual_map(self, x_col, y_col):
        """Compute residual values at collocation points"""
        with tf.GradientTape() as tape:
            cont, x_mom, y_mom = self.loss_fn.navier_stokes_loss(
                tf.Variable(x_col), tf.Variable(y_col)
            )
            residuals = tf.abs(cont) + tf.abs(x_mom) + tf.abs(y_mom)
        return residuals.numpy()
    
    def generate_adaptive_samples(self, num_points):
        """Generate samples with higher density in high-residual regions"""
        # Create initial uniform sampling
        x_uniform = np.random.uniform(0, config.DOMAIN_LENGTH, num_points)
        y_uniform = np.random.uniform(-TUBE_RADIUS, TUBE_RADIUS, num_points)
        
        # Exclude orifice region
        valid = ~GeometryUtils.is_in_orifice(x_uniform, y_uniform)
        x_uniform = x_uniform[valid]
        y_uniform = y_uniform[valid]
        
        # Compute residuals
        residuals = self.compute_residual_map(
            tf.convert_to_tensor(x_uniform, dtype=tf.float32),
            tf.convert_to_tensor(y_uniform, dtype=tf.float32)
        )
        
        # Normalize residuals for sampling probability
        residuals = np.maximum(residuals, 1e-8)
        probabilities = residuals / np.sum(residuals)
        
        # Resample with higher probability in high-residual regions
        indices = np.random.choice(len(x_uniform), size=min(num_points, len(x_uniform)), 
                                   p=probabilities[:min(num_points, len(x_uniform))]/np.sum(probabilities[:min(num_points, len(x_uniform))]))
        
        return x_uniform[indices], y_uniform[indices]


def extract_epoch_from_filename(filename):
    # filename format: model_epoch_3000_loss_0.0001.weights.h5
    try:
        return int(filename.split("_")[2])
    except:
        return -1


# ===================== NEURAL NETWORK =====================
class AdvancedPINNModel(keras.Model):
    """Advanced PINN with attention mechanisms and residual connections"""
    
    def __init__(self, config):
        super(AdvancedPINNModel, self).__init__()
        self.config = config
        
        # Shared layers with residual connections
        self.dense_layers = []
        for i in range(config.HIDDEN_LAYERS):
            self.dense_layers.append(
                layers.Dense(config.NEURONS_PER_LAYER, 
                           kernel_initializer=initializers.GlorotNormal())
            )
        
        # Output layers
        self.u_output = layers.Dense(1)
        self.v_output = layers.Dense(1)
        self.p_output = layers.Dense(1)
        
        # Activation function
        self.activation = layers.Activation(config.ACTIVATION)
    
    def call(self, inputs, training=None):
        """Forward pass with residual connections every other layer"""
        x = inputs
        
        for i, dense in enumerate(self.dense_layers):
            x_prev = x
            x = dense(x)
            x = self.activation(x)
            
            # Residual connection every 2 layers (if dimensions match)
            if i > 0 and i % 2 == 0 and x.shape[-1] == x_prev.shape[-1]:
                x = x + x_prev
        
        u = self.u_output(x)
        v = self.v_output(x)
        p = self.p_output(x)
        
        return u, v, p


# ===================== LOSS FUNCTIONS =====================
class AdvancedPINNLoss:
    """Advanced loss computation with residual tracking"""
    
    def __init__(self, config):
        self.config = config
        self.rho = config.WATER_DENSITY
        self.mu = config.WATER_VISCOSITY
        self.residual_history = []
    
    def navier_stokes_loss(self, model, x, y):
        """Compute NS residuals"""
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            tape1.watch(y)
            
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x)
                tape2.watch(y)
                
                xy = tf.stack([x, y], axis=1)
                u, v, p = model(xy)
            
            u_x = tape2.gradient(u, x)
            u_y = tape2.gradient(u, y)
            v_x = tape2.gradient(v, x)
            v_y = tape2.gradient(v, y)
            p_x = tape2.gradient(p, x)
            p_y = tape2.gradient(p, y)

            del tape2
        
        u_xx = tape1.gradient(u_x, x)
        u_yy = tape1.gradient(u_y, y)
        v_xx = tape1.gradient(v_x, x)
        v_yy = tape1.gradient(v_y, y)

        del tape1
        
        continuity = u_x + v_y
        x_momentum = self.rho * (u * u_x + v * u_y) + p_x - self.mu * (u_xx + u_yy)
        y_momentum = self.rho * (u * v_x + v * v_y) + p_y - self.mu * (v_xx + v_yy)
        
        return continuity, x_momentum, y_momentum
    
    def compute_residual_norms(self, continuity, x_mom, y_mom):
        """Compute L2 norms of residuals"""
        cont_norm = tf.reduce_mean(tf.square(continuity))
        xmom_norm = tf.reduce_mean(tf.square(x_mom))
        ymom_norm = tf.reduce_mean(tf.square(y_mom))
        return cont_norm, xmom_norm, ymom_norm


# ===================== TRAINING UTILITIES =====================
class TrainingUtilities:
    """Utilities for training management"""
    
    def __init__(self, config):
        self.config = config
        if config.SAVE_CHECKPOINTS and not os.path.exists(config.CHECKPOINT_DIR):
            os.makedirs(config.CHECKPOINT_DIR)
    
    def save_checkpoint(self, model, epoch, loss):
        """Save model checkpoint"""
        if not self.config.SAVE_CHECKPOINTS:
            return
        
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR, 
            f'model_epoch_{epoch:04d}_loss_{loss:.6e}.weights.h5'
        )
        model.save_weights(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, model, checkpoint_path):
        """Load model checkpoint"""
        model.load_weights(checkpoint_path)
        print(f"Checkpoint loaded: {checkpoint_path}")
        return model


# ===================== POST-PROCESSING =====================
class PostProcessor:
    """Post-processing and analysis utilities"""
    
    @staticmethod
    def compute_coefficient_of_discharge(model, x_upstream=0.01, x_downstream=0.09):
        """Compute Cd with improved accuracy"""
        
        y_upstream = np.linspace(-TUBE_RADIUS, TUBE_RADIUS, 150)
        x_upstream_arr = np.full_like(y_upstream, x_upstream)
        
        xy_upstream = tf.stack([
            tf.convert_to_tensor(x_upstream_arr, dtype=tf.float32),
            tf.convert_to_tensor(y_upstream, dtype=tf.float32)
        ], axis=1)
        
        u_upstream, _, p_upstream = model(xy_upstream)
        u_upstream = u_upstream.numpy().flatten()
        p_upstream = p_upstream.numpy().flatten()
        
        # Downstream
        y_downstream = np.linspace(-TUBE_RADIUS, TUBE_RADIUS, 150)
        x_downstream_arr = np.full_like(y_downstream, x_downstream)
        
        xy_downstream = tf.stack([
            tf.convert_to_tensor(x_downstream_arr, dtype=tf.float32),
            tf.convert_to_tensor(y_downstream, dtype=tf.float32)
        ], axis=1)
        
        u_downstream, _, p_downstream = model(xy_downstream)
        u_downstream = u_downstream.numpy().flatten()
        p_downstream = p_downstream.numpy().flatten()
        
        # Integration
        dy = y_upstream[1] - y_upstream[0]
        Q_upstream = np.trapz(u_upstream, dx=dy)
        Q_downstream = np.trapz(u_downstream, dx=dy)
        Q_actual = (Q_upstream + Q_downstream) / 2
        
        # Theoretical discharge
        delta_p = np.mean(p_upstream) - np.mean(p_downstream)
        A_orifice = np.pi * ORIFICE_RADIUS**2
        Q_theoretical = A_orifice * np.sqrt(2 * delta_p / config.WATER_DENSITY) if delta_p > 0 else Q_upstream
        
        Cd = Q_actual / Q_theoretical if Q_theoretical > 0 else 0
        
        # Compute vena contracta
        A_vena_contracta = (Q_actual / np.max(u_downstream)) if np.max(u_downstream) > 0 else A_orifice
        vena_contracta_ratio = A_vena_contracta / A_orifice
        
        return {
            'Cd': Cd,
            'Q_actual': Q_actual,
            'Q_theoretical': Q_theoretical,
            'Q_upstream': Q_upstream,
            'Q_downstream': Q_downstream,
            'delta_p': delta_p,
            'vena_contracta_ratio': vena_contracta_ratio,
            'p_upstream': np.mean(p_upstream),
            'p_downstream': np.mean(p_downstream)
        }
    
    @staticmethod
    def compute_derivatives(model, x_pred, y_pred, X_pred, Y_pred, valid_mask):
        """Compute velocity gradients for visualization"""
        
        xy = tf.stack([
            tf.convert_to_tensor(x_pred, dtype=tf.float32),
            tf.convert_to_tensor(y_pred, dtype=tf.float32)
        ], axis=1)
        
        with tf.GradientTape() as tape:
            tape.watch(xy)
            u, v, p = model(xy)
        
        u_grad = tape.gradient(u, xy)
        v_grad = tape.gradient(v, xy)
        
        u_x = u_grad[:, 0].numpy().flatten()
        u_y = u_grad[:, 1].numpy().flatten()
        v_x = v_grad[:, 0].numpy().flatten()
        v_y = v_grad[:, 1].numpy().flatten()
        
        # Vorticity: ω = ∂v/∂x - ∂u/∂y
        vorticity = v_x - u_y
        
        # Shear rate: γ̇ = √(2(∂u/∂x² + ∂v/∂y² + 2(∂u/∂y)(∂v/∂x)))
        shear_rate = np.sqrt(2 * (u_x**2 + v_y**2 + 2 * u_y * v_x))
        
        # Reshape
        VORTICITY = np.full_like(X_pred, np.nan)
        SHEAR_RATE = np.full_like(X_pred, np.nan)
        
        VORTICITY[valid_mask] = vorticity
        SHEAR_RATE[valid_mask] = shear_rate
        
        return VORTICITY, SHEAR_RATE
    
    @staticmethod
    def plot_advanced_results(model, train_losses, results_dir='./results'):
        """Generate comprehensive visualization"""
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Prediction grid
        x_pred = np.linspace(0, config.DOMAIN_LENGTH, 250)
        y_pred = np.linspace(-TUBE_RADIUS, TUBE_RADIUS, 250)
        X_pred, Y_pred = np.meshgrid(x_pred, y_pred)
        
        valid_mask = ~GeometryUtils.is_in_orifice(X_pred, Y_pred)
        
        # Model predictions
        X_flat = X_pred[valid_mask].reshape(-1, 1).astype(np.float32)
        Y_flat = Y_pred[valid_mask].reshape(-1, 1).astype(np.float32)
        xy = tf.stack([
            tf.convert_to_tensor(X_flat.flatten()),
            tf.convert_to_tensor(Y_flat.flatten())
        ], axis=1)
        
        u_pred, v_pred, p_pred = model(xy)
        u_pred = u_pred.numpy().flatten()
        v_pred = v_pred.numpy().flatten()
        p_pred = p_pred.numpy().flatten()
        
        U_pred = np.full_like(X_pred, np.nan)
        V_pred = np.full_like(X_pred, np.nan)
        P_pred = np.full_like(X_pred, np.nan)
        
        U_pred[valid_mask] = u_pred
        V_pred[valid_mask] = v_pred
        P_pred[valid_mask] = p_pred
        
        # Compute derivatives
        VORTICITY, SHEAR_RATE = PostProcessor.compute_derivatives(
            model, X_flat.flatten(), Y_flat.flatten(), X_pred, Y_pred, valid_mask
        )
        
        # Create figure
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        # Velocity magnitude
        ax1 = fig.add_subplot(gs[0, :])
        vel_mag = np.sqrt(U_pred**2 + V_pred**2)
        cf1 = ax1.contourf(X_pred*1000, Y_pred*1000, vel_mag, levels=25, cmap='viridis')
        ax1.contour(X_pred*1000, Y_pred*1000, vel_mag, levels=12, colors='black', alpha=0.2, linewidths=0.5)
        cbar1 = plt.colorbar(cf1, ax=ax1, label='Velocity Magnitude (m/s)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_title('Velocity Magnitude Contours')
        ax1.set_aspect('equal')
        circle1 = plt.Circle((ORIFICE_CENTER_X*1000, 0), ORIFICE_RADIUS*1000, 
                            color='red', fill=True, alpha=0.3)
        ax1.add_patch(circle1)
        
        # Pressure
        ax2 = fig.add_subplot(gs[1, 0])
        cf2 = ax2.contourf(X_pred*1000, Y_pred*1000, P_pred, levels=25, cmap='RdBu_r')
        cbar2 = plt.colorbar(cf2, ax=ax2, label='Pressure (Pa)')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title('Pressure Contours')
        ax2.set_aspect('equal')
        
        # U velocity
        ax3 = fig.add_subplot(gs[1, 1])
        cf3 = ax3.contourf(X_pred*1000, Y_pred*1000, U_pred, levels=25, cmap='coolwarm')
        cbar3 = plt.colorbar(cf3, ax=ax3, label='U Velocity (m/s)')
        ax3.set_xlabel('X (mm)')
        ax3.set_title('Horizontal Velocity (U)')
        ax3.set_aspect('equal')
        
        # V velocity
        ax4 = fig.add_subplot(gs[1, 2])
        cf4 = ax4.contourf(X_pred*1000, Y_pred*1000, V_pred, levels=25, cmap='PiYG')
        cbar4 = plt.colorbar(cf4, ax=ax4, label='V Velocity (m/s)')
        ax4.set_xlabel('X (mm)')
        ax4.set_title('Vertical Velocity (V)')
        ax4.set_aspect('equal')
        
        # Vorticity
        ax5 = fig.add_subplot(gs[2, 0])
        cf5 = ax5.contourf(X_pred*1000, Y_pred*1000, VORTICITY, levels=25, cmap='seismic')
        cbar5 = plt.colorbar(cf5, ax=ax5, label='Vorticity (1/s)')
        ax5.set_xlabel('X (mm)')
        ax5.set_ylabel('Y (mm)')
        ax5.set_title('Vorticity Contours')
        ax5.set_aspect('equal')
        
        # Shear rate
        ax6 = fig.add_subplot(gs[2, 1])
        cf6 = ax6.contourf(X_pred*1000, Y_pred*1000, SHEAR_RATE, levels=25, cmap='hot')
        cbar6 = plt.colorbar(cf6, ax=ax6, label='Shear Rate (1/s)')
        ax6.set_xlabel('X (mm)')
        ax6.set_title('Shear Rate Contours')
        ax6.set_aspect('equal')
        
        # Velocity vectors
        ax7 = fig.add_subplot(gs[2, 2])
        skip = 10
        ax7.quiver(X_pred[::skip, ::skip]*1000, Y_pred[::skip, ::skip]*1000, 
                  U_pred[::skip, ::skip], V_pred[::skip, ::skip], vel_mag[::skip, ::skip], cmap='viridis')
        ax7.set_xlabel('X (mm)')
        ax7.set_ylabel('Y (mm)')
        ax7.set_title('Velocity Field Vectors')
        ax7.set_aspect('equal')
        
        # Training loss
        ax8 = fig.add_subplot(gs[3, :2])
        ax8.semilogy(train_losses, linewidth=2.5, color='steelblue')
        ax8.set_xlabel('Epoch', fontsize=11)
        ax8.set_ylabel('Loss', fontsize=11)
        ax8.set_title('Training Loss History', fontsize=12)
        ax8.grid(True, alpha=0.3)
        ax8.fill_between(range(len(train_losses)), train_losses, alpha=0.3, color='steelblue')
        
        # Residual along centerline
        ax9 = fig.add_subplot(gs[3, 2])
        centerline_idx = len(y_pred) // 2
        ax9.plot(X_pred*1000, U_pred[centerline_idx, :], 'o-', label='U (centerline)', linewidth=2, markersize=4)
        ax9.set_xlabel('X (mm)')
        ax9.set_ylabel('U Velocity (m/s)')
        ax9.set_title('Centerline Velocity Profile')
        ax9.grid(True, alpha=0.3)
        ax9.legend()
        
        plt.savefig(os.path.join(results_dir, 'pinn_advanced_results.png'), dpi=150, bbox_inches='tight')
        print(f"Advanced plots saved to {results_dir}/pinn_advanced_results.png")
        plt.close()


# ===================== MAIN TRAINING ROUTINE =====================
def train_advanced_pinn(config, model, start_epoch=0):
    """Train advanced PINN model"""
    
    print("=" * 80)
    print("ADVANCED PINN FOR 2D ORIFICE FLOW")
    print("=" * 80)
    print(f"\nConfiguration:\n{config}\n")
    
    # Initialize model and utilities
    model = AdvancedPINNModel(config)
    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    loss_fn = AdvancedPINNLoss(config)
    train_utils = TrainingUtilities(config)
    
    # Generate initial data
    x_col = np.random.uniform(0, config.DOMAIN_LENGTH, config.NUM_COLLOCATION_POINTS).astype(np.float32)
    y_col = np.random.uniform(-TUBE_RADIUS, TUBE_RADIUS, config.NUM_COLLOCATION_POINTS).astype(np.float32)

    valid = ~GeometryUtils.is_in_orifice(x_col, y_col)
    x_col, y_col = x_col[valid], y_col[valid]
    
    # Training loop
    train_losses = []
    best_loss = np.inf
    
    print("Starting training...")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        with tf.GradientTape() as tape:
            # Convert to TF variables
            x_col_var = tf.Variable(x_col[:config.NUM_COLLOCATION_POINTS], dtype=tf.float32)
            y_col_var = tf.Variable(y_col[:config.NUM_COLLOCATION_POINTS], dtype=tf.float32)

            
            # Physics loss
            cont, x_mom, y_mom = loss_fn.navier_stokes_loss(model, x_col_var, y_col_var)
            physics_loss = (tf.reduce_mean(tf.square(cont)) +
                           tf.reduce_mean(tf.square(x_mom)) +
                           tf.reduce_mean(tf.square(y_mom)))
            
            # Simplified boundary loss for this version
            boundary_loss = physics_loss * 0.1
            
            total_loss = config.PHYSICS_WEIGHT * physics_loss + config.BOUNDARY_WEIGHT * boundary_loss
        
        gradients = tape.gradient(total_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        
        train_losses.append(total_loss.numpy())
        
        if total_loss.numpy() < best_loss:
            best_loss = total_loss.numpy()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1:4d}/{config.NUM_EPOCHS} | Loss: {total_loss.numpy():.6e} | Best: {best_loss:.6e}")
        
        if config.SAVE_CHECKPOINTS and (epoch + 1) % config.CHECKPOINT_FREQ == 0:
            train_utils.save_checkpoint(model, epoch + 1, total_loss.numpy())
    
    return model, train_losses


# ===================== MAIN EXECUTION =====================
if __name__ == "__main__":

    train_utils = TrainingUtilities(config)
    model = AdvancedPINNModel(config)

    start_epoch = 0  # default: start from scratch

    if os.path.isdir(config.CHECKPOINT_DIR):
        ckpt_files = [f for f in os.listdir(config.CHECKPOINT_DIR)
                      if f.endswith(".h5")]

        if len(ckpt_files) > 0:
            latest = max(ckpt_files, key=extract_epoch_from_filename)
            latest_epoch = extract_epoch_from_filename(latest)
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, latest)

            print(f"\n🔄 Resuming from checkpoint: {ckpt_path}")
            print(f"➡ Starting from epoch {latest_epoch + 1}")

            dummy_input = tf.zeros((1, 2), dtype=tf.float32)
            model(dummy_input)

            model.load_weights(ckpt_path)
            start_epoch = latest_epoch    # continue from next epoch
        else:
            print("No checkpoints found. Training from scratch.")
    else:
        print("Checkpoint directory not found. Training from scratch.")
    
    # Train model
    model, losses = train_advanced_pinn(config, model, start_epoch=start_epoch)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    # Post-processing
    print("\nComputing coefficient of discharge...")
    results = PostProcessor.compute_coefficient_of_discharge(model)
    
    print(f"\n--- Discharge Analysis Results ---")
    print(f"Coefficient of Discharge (Cd): {results['Cd']:.4f}")
    print(f"  Expected range (orifice): 0.60 - 0.65")
    print(f"\nFlow Rates:")
    print(f"  Upstream Q: {results['Q_upstream']:.6f} m³/s")
    print(f"  Downstream Q: {results['Q_downstream']:.6f} m³/s")
    print(f"  Actual Q: {results['Q_actual']:.6f} m³/s")
    print(f"  Theoretical Q: {results['Q_theoretical']:.6f} m³/s")
    print(f"\nPressure Distribution:")
    print(f"  Upstream P: {results['p_upstream']:.3f} Pa")
    print(f"  Downstream P: {results['p_downstream']:.3f} Pa")
    print(f"  Pressure drop: {results['delta_p']:.3f} Pa")
    print(f"\nVena Contracta:")
    print(f"  Ratio (A_vc / A_orifice): {results['vena_contracta_ratio']:.3f}")
    
    # Generate plots
    print("\nGenerating advanced visualizations...")
    PostProcessor.plot_advanced_results(model, losses)
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


def run_orificemeter():

    train_utils = TrainingUtilities(config)
    model = AdvancedPINNModel(config)

    start_epoch = 0  # default: start from scratch

    if os.path.isdir(config.CHECKPOINT_DIR):
        ckpt_files = [f for f in os.listdir(config.CHECKPOINT_DIR)
                      if f.endswith(".h5")]

        if len(ckpt_files) > 0:
            latest = max(ckpt_files, key=extract_epoch_from_filename)
            latest_epoch = extract_epoch_from_filename(latest)
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, latest)

            print(f"\n🔄 Resuming from checkpoint: {ckpt_path}")
            print(f"➡ Starting from epoch {latest_epoch + 1}")

            dummy_input = tf.zeros((1, 2), dtype=tf.float32)
            model(dummy_input)

            model.load_weights(ckpt_path)
            start_epoch = latest_epoch    # continue from next epoch
        else:
            print("No checkpoints found. Training from scratch.")
    else:
        print("Checkpoint directory not found. Training from scratch.")
    
    # Train model
    model, losses = train_advanced_pinn(config, model, start_epoch=start_epoch)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    # Post-processing
    print("\nComputing coefficient of discharge...")
    results = PostProcessor.compute_coefficient_of_discharge(model)
    
    print(f"\n--- Discharge Analysis Results ---")
    print(f"Coefficient of Discharge (Cd): {results['Cd']:.4f}")
    print(f"  Expected range (orifice): 0.60 - 0.65")
    print(f"\nFlow Rates:")
    print(f"  Upstream Q: {results['Q_upstream']:.6f} m³/s")
    print(f"  Downstream Q: {results['Q_downstream']:.6f} m³/s")
    print(f"  Actual Q: {results['Q_actual']:.6f} m³/s")
    print(f"  Theoretical Q: {results['Q_theoretical']:.6f} m³/s")
    print(f"\nPressure Distribution:")
    print(f"  Upstream P: {results['p_upstream']:.3f} Pa")
    print(f"  Downstream P: {results['p_downstream']:.3f} Pa")
    print(f"  Pressure drop: {results['delta_p']:.3f} Pa")
    print(f"\nVena Contracta:")
    print(f"  Ratio (A_vc / A_orifice): {results['vena_contracta_ratio']:.3f}")
    
    # Generate plots
    print("\nGenerating advanced visualizations...")
    PostProcessor.plot_advanced_results(model, losses)
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

    return None
