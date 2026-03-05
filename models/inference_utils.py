"""
Inference utilities for scratch models (DNN and CNN)
Extracted from notebooks for Streamlit deployment
"""

import numpy as np

# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================

def relu(Z):
    """ReLU activation function"""
    return np.maximum(0, Z)


def softmax(Z):
    """Softmax activation with numerical stability"""
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def z_func(W, X, b):
    """Linear transformation: Z = WX + b"""
    return np.dot(W, X) + b


def batch_normalization_forward(Z, gamma, beta, r_mean, r_var, training=False, epsilon=1e-7, momentum=0.9):
    """
    Batch normalization forward pass
    For inference (training=False), use running statistics
    """
    if training:
        m = np.mean(Z, axis=1, keepdims=True)
        v = np.var(Z, axis=1, keepdims=True)
        z_n = (Z - m) / np.sqrt(v + epsilon)
        r_mean = (momentum * r_mean) + (1 - momentum) * m
        r_var = (momentum * r_var) + (1 - momentum) * v
        cache = {"z_n": z_n, "m": m, "v": v, "gamma": gamma, "beta": beta, 
                 "r_mean": r_mean, "r_var": r_var}
    else:
        z_n = (Z - r_mean) / np.sqrt(r_var + epsilon)
        cache = None

    z_t = (gamma * z_n) + beta
    return z_t, cache


# ============================================================================
# DNN FORWARD PROPAGATION
# ============================================================================

def forward_propagation_dnn(parameters, bn_parameters, bn_stats, X, keep_prob=1.0, training=False):
    """
    Forward propagation for DNN with BatchNorm
    For inference, set training=False and keep_prob=1.0
    """
    if not training:
        keep_prob = 1.0
   
    # Layer 1: Dense(512) + BN + ReLU + Dropout
    z_1 = z_func(parameters['W1'], X, parameters['b1'])
    z_t_1, cache_1 = batch_normalization_forward(
        z_1, bn_parameters["gamma1"], bn_parameters["beta1"], 
        bn_stats["running_mean1"], bn_stats["running_var1"], 
        training, epsilon=1e-7
    )
    a_1 = relu(z_t_1)
    if training:
        d_1 = np.random.rand(a_1.shape[0], a_1.shape[1]) < keep_prob
        a_1 *= d_1 
        a_1 /= keep_prob
    
    # Layer 2: Dense(256) + BN + ReLU + Dropout
    z_2 = z_func(parameters['W2'], a_1, parameters['b2'])
    z_t_2, cache_2 = batch_normalization_forward(
        z_2, bn_parameters["gamma2"], bn_parameters["beta2"], 
        bn_stats["running_mean2"], bn_stats["running_var2"], 
        training, epsilon=1e-7
    )
    a_2 = relu(z_t_2)
    if training:
        d_2 = np.random.rand(a_2.shape[0], a_2.shape[1]) < keep_prob
        a_2 *= d_2 
        a_2 /= keep_prob

    # Layer 3: Dense(128) + BN + ReLU + Dropout
    z_3 = z_func(parameters['W3'], a_2, parameters['b3'])
    z_t_3, cache_3 = batch_normalization_forward(
        z_3, bn_parameters["gamma3"], bn_parameters["beta3"], 
        bn_stats["running_mean3"], bn_stats["running_var3"], 
        training, epsilon=1e-7
    )
    a_3 = relu(z_t_3)
    if training:
        d_3 = np.random.rand(a_3.shape[0], a_3.shape[1]) < keep_prob
        a_3 *= d_3 
        a_3 /= keep_prob

    # Layer 4: Dense(10) + Softmax (output layer)
    z_4 = z_func(parameters['W4'], a_3, parameters['b4'])
    a_4 = softmax(z_4)

    return a_4  


def predict_dnn(model_data, X):
    """
    Predict class for a single image using DNN.
    X is assumed to be already flattened (12288, 1) and normalized in app.py
    """
    probs = forward_propagation_dnn(
        model_data['parameters'],
        model_data['bn_parameters'],
        model_data['bn_stats'],
        X,
        keep_prob=1.0,
        training=False
    )
    
    predicted_class = np.argmax(probs, axis=0)[0]
    probabilities = probs.flatten()
    
    return int(predicted_class), probabilities


# ============================================================================
# CNN HELPER FUNCTIONS
# ============================================================================

def batchnorm_forward_cnn(Z, gamma, beta, running_mean, running_var, training=False, epsilon=1e-5):
    """Batch normalization for CNN (operates on spatial dimensions)"""
    if training:
        mu = np.mean(Z, axis=(0, 1, 2), keepdims=True)
        var = np.var(Z, axis=(0, 1, 2), keepdims=True)
        Z_norm = (Z - mu) / np.sqrt(var + epsilon)
        cache = (Z, Z_norm, mu, var, gamma, beta, epsilon)
    else:
        Z_norm = (Z - running_mean) / np.sqrt(running_var + epsilon)
        cache = None
    
    Z_out = gamma * Z_norm + beta
    return Z_out, cache


def conv_forward_simple(X, W, b, stride, pad):
    """Simplified conv forward for inference"""
    # FIX: Renamed H and W to img_H and img_W to avoid overwriting the W (weights) parameter
    m, img_H, img_W, C_in = X.shape
    f, _, _, C_out = W.shape
    
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant')
    
    H_out = (img_H + 2*pad - f) // stride + 1
    W_out = (img_W + 2*pad - f) // stride + 1
    
    Z = np.zeros((m, H_out, W_out, C_out))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            X_slice = X_pad[:, h_start:h_start+f, w_start:w_start+f, :]
            
            for k in range(C_out):
                Z[:, i, j, k] = np.sum(X_slice * W[:, :, :, k], axis=(1, 2, 3)) + b[0, 0, 0, k]
    
    return Z


def pool_forward_simple(X, pool_size, stride, mode='max'):
    """Simplified pooling for inference"""
    m, img_H, img_W, C = X.shape
    H_out = (img_H - pool_size) // stride + 1
    W_out = (img_W - pool_size) // stride + 1
    
    Z = np.zeros((m, H_out, W_out, C))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            X_slice = X[:, h_start:h_start+pool_size, w_start:w_start+pool_size, :]
            
            if mode == 'max':
                Z[:, i, j, :] = np.max(X_slice, axis=(1, 2))
            else:
                Z[:, i, j, :] = np.mean(X_slice, axis=(1, 2))
    
    return Z


# ============================================================================
# CNN FORWARD PROPAGATION
# ============================================================================

def forward_propagation_cnn(X, architecture, parameters, training=False):
    """Forward propagation for CNN"""
    A = X
    param_idx = 1
    bn_idx = 1
    
    for layer in architecture:
        if layer["type"] == "conv":
            W = parameters[f"W{param_idx}"]
            b = parameters[f"b{param_idx}"]
            A = conv_forward_simple(A, W, b, layer["stride"], layer["pad"])
            param_idx += 1
        
        elif layer["type"] == "batchnorm":
            gamma = parameters[f"gamma{bn_idx}"]
            beta = parameters[f"beta{bn_idx}"]
            running_mean = parameters[f"running_mean{bn_idx}"]
            running_var = parameters[f"running_var{bn_idx}"]
            A, _ = batchnorm_forward_cnn(A, gamma, beta, running_mean, running_var, training)
            bn_idx += 1
        
        elif layer["type"] == "relu":
            A = relu(A)
        
        elif layer["type"] == "pool":
            A = pool_forward_simple(A, layer["pool_size"], layer["stride"], layer.get("mode", "max"))
        
        elif layer["type"] == "dropout":
            pass
        
        elif layer["type"] == "flatten":
            A = A.reshape(A.shape[0], -1).T
        
        elif layer["type"] == "dense":
            W = parameters[f"W{param_idx}"]
            b = parameters[f"b{param_idx}"]
            A = z_func(W, A, b)
            param_idx += 1
        
        elif layer["type"] == "softmax":
            A = softmax(A)
    
    return A


def predict_cnn(model_data, X):
    """
    Predict class for a single image using CNN
    X is assumed to be batched (1, 64, 64, 3) and normalized in app.py
    """
    probs = forward_propagation_cnn(
        X,
        model_data['architecture'],
        model_data['parameters'],
        training=False
    )
    
    predicted_class = np.argmax(probs, axis=0)[0]
    probabilities = probs.flatten()
    
    return int(predicted_class), probabilities