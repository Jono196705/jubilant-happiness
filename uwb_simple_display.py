"""
Simple UWB Position Display
A clean, simplified version that works with the HTML interface.
Optimized for accuracy and speed.
"""
import time
import json
import math
import random
import threading
import socket
import struct
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from collections import deque
import gc

app = Flask(__name__)
CORS(app)

# Configuration
UDP_PORT = 8000
ARTNET_IP = "127.0.0.1"  # Localhost - for local Art-Net software
# Alternative options:
# ARTNET_IP = "10.10.0.65"  # Direct to light fixture IP
# ARTNET_IP = "255.255.255.255"  # Broadcast to all devices on network
ARTNET_UNIVERSE = 0
ARTNET_PORT = 6454

PAN_CHANNEL = 1  # DMX channel number for pan (1-based)
TILT_CHANNEL = 2  # DMX channel number for tilt (1-based)
SPEED_CHANNEL = None  # Optional pan/tilt speed channel (fixture-dependent)
SPEED_VALUE = 0       # 0 often means fastest; check your fixture manual
DIMMER_CHANNEL = None # Optional dimmer/intensity channel
DIMMER_VALUE = 255    # Full brightness by default

# Global variables
current_ranges = {"1785": 0.0, "1786": 0.0}
distance_a1_a2 = 1.5

# Map incoming anchor IDs from hardware to internal IDs used in calculations
# Your stream currently uses 1783 and 1782; map them to 1786 and 1785 respectively
ANCHOR_ID_MAP = {
    "1782": "1785",  # left anchor
    "1783": "1786"   # right anchor
}

# Anchor positions (for automatic distance calculation)
anchor_positions = {
    "1785": {"x": -0.75, "y": 0.0, "z": 0.0},  # Left anchor
    "1786": {"x": 0.75, "y": 0.0, "z": 0.0}    # Right anchor
}

# UWB data connection
uwb_socket = None
uwb_data = None

# Art-Net connection
artnet_socket = None
# DMX send timestamp for rate gating
last_dmx_send_time = 0.0

# Demo mode
DEMO_ENABLED = True  # Enable demo mode for testing without UWB hardware
demo_mode = False
demo_time = 0.0

# Calibration - REASONABLE VALUES FOR PROPER DMX OUTPUT
pan_offset = 0.0
tilt_offset = 0.0
pan_scale = 1.0  # Reasonable sensitivity
tilt_scale = 1.0  # Reasonable sensitivity
center_offset_x = 0.0
center_offset_y = 0.0
spherical_scale = 1.0  # Reasonable overall sensitivity
reference_distance = 3.0
smoothing_factor = 0.0  # No smoothing for instant response
light_position_x = 0.0
light_position_y = 0.0  # Center between anchors (not at front of room)
light_height = 2.0  # Height of light fixture above tracking plane (adjustable) - typical ceiling height

# Extended range calibration parameters
pan_range_scale = 1.0  # Scale factor for pan range (1.0 = normal, 2.0 = double range)
tilt_range_scale = 1.0  # Scale factor for tilt range (1.0 = normal, 2.0 = double range)
pan_range_offset = 0.0  # Offset for pan range center
tilt_range_offset = 0.0  # Offset for tilt range center
pan_min_angle = -180.0  # Minimum pan angle in degrees
pan_max_angle = 180.0   # Maximum pan angle in degrees
tilt_min_angle = -90.0  # Minimum tilt angle in degrees
tilt_max_angle = 90.0   # Maximum tilt angle in degrees

# Movement smoothing and filtering
last_pan_value = 128
last_tilt_value = 128

# Position filtering for accuracy - ULTRA INSTANT RESPONSE
position_history = deque(maxlen=1)   # No history for instant response
range_history = deque(maxlen=1)      # No history for instant response
velocity_filter = 0.0                # No filtering for instant response

# Performance optimization - ULTRA INSTANT RESPONSE
last_position_time = 0
position_update_rate = 0.0           # No delay - instant updates
dmx_update_rate = 0.0                # No delay - instant DMX

# Calibration file
CALIBRATION_FILE = 'calibration.json'

# Auto-calibration state
auto_calibration_mode = False
calibration_samples = deque(maxlen=100)  # Limit to 100 samples
calibration_step = 0

# Point and learn calibration state
point_learn_mode = False
point_learn_samples = deque(maxlen=50)  # Limit to 50 samples

# Position tracking for status
last_x = 0.0
last_y = 0.0
last_position_time = time.time()
point_learn_step = 0

# Enhanced room configuration
room_config = {
    'width': 1.9,
    'length': 2.5,
    'light_height': 0.0,
    'anchor_distance': 1.9,
    'tag_height': 0.5,  # Height of UWB tag above ground (adjustable as needed)
    'light_orientation': 'floor'  # 'floor' for upright, 'ceiling' for hanging
}
room_setup_complete = False

# Manual mode flag to disable automatic DMX sending
manual_mode = False
auto_follow_mode = True  # Enable automatic following by default
current_manual_pan = 128
current_manual_tilt = 128

# Room follow / mapping flags
room_follow_mode_active = False
room_mapping_autofollow_enabled = True

# Kalman Filter Configuration
kalman_enabled = False  # Can be toggled on/off
kalman_process_noise = 0.01  # Process noise (how much we trust the model)
kalman_measurement_noise = 0.1  # Measurement noise (how much we trust the sensor)
kalman_initial_uncertainty = 1.0  # Initial uncertainty

# Motion prediction state
velocity_x = 0.0
velocity_y = 0.0
last_update_time = time.time()

class LightweightKalmanFilter:
    """Lightweight Kalman filter for UWB range and position filtering"""
    
    def __init__(self, process_noise=0.001, measurement_noise=0.05, initial_uncertainty=0.5):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.uncertainty = initial_uncertainty
        self.estimate = None
        self.velocity = 0.0
        self.last_time = None
        
    def update(self, measurement, timestamp=None):
        """Update filter with new measurement"""
        if timestamp is None:
            timestamp = time.time()
            
        if self.estimate is None:
            # First measurement - initialize
            self.estimate = measurement
            self.last_time = timestamp
            return measurement
            
        # Calculate time delta
        dt = timestamp - self.last_time
        if dt <= 0:
            return self.estimate
            
        # Prediction step
        predicted_estimate = self.estimate + self.velocity * dt
        predicted_uncertainty = self.uncertainty + self.process_noise * dt
        
        # Update step (measurement fusion)
        kalman_gain = predicted_uncertainty / (predicted_uncertainty + self.measurement_noise)
        
        # Update estimate
        self.estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        
        # Update uncertainty
        self.uncertainty = (1 - kalman_gain) * predicted_uncertainty
        
        # Update velocity (simple finite difference)
        if dt > 0:
            self.velocity = (measurement - self.estimate) / dt
            
        self.last_time = timestamp
        return self.estimate
        
    def reset(self):
        """Reset filter state"""
        self.estimate = None
        self.velocity = 0.0
        self.uncertainty = self.initial_uncertainty
        self.last_time = None
        
    def get_state(self):
        """Get current filter state for debugging"""
        return {
            'estimate': self.estimate,
            'velocity': self.velocity,
            'uncertainty': self.uncertainty,
            'last_time': self.last_time
        }

# Initialize Kalman filters for each anchor
kalman_filters = {
    "1785": LightweightKalmanFilter(
        process_noise=kalman_process_noise,
        measurement_noise=kalman_measurement_noise,
        initial_uncertainty=kalman_initial_uncertainty
    ),
    "1786": LightweightKalmanFilter(
        process_noise=kalman_process_noise,
        measurement_noise=kalman_measurement_noise,
        initial_uncertainty=kalman_initial_uncertainty
    )
}

# Kalman filter for position (x, y coordinates)
position_kalman = LightweightKalmanFilter(
    process_noise=0.02,  # Slightly higher for position
    measurement_noise=0.15,  # Higher for position measurements
    initial_uncertainty=1.0
)

def load_calibration():
    """Load calibration from file - FORCED TO MAXIMUM SPEED"""
    global pan_offset, tilt_offset, pan_scale, tilt_scale, center_offset_x, center_offset_y
    global spherical_scale, reference_distance, smoothing_factor, light_position_x, light_position_y
    global PAN_CHANNEL, TILT_CHANNEL, distance_a1_a2, anchor_positions, light_height
    global pan_range_scale, tilt_range_scale, pan_range_offset, tilt_range_offset
    global pan_min_angle, pan_max_angle, tilt_min_angle, tilt_max_angle
    
    # Load calibration values from file (don't force maximum)
    # These will be overridden by saved values below
    
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                data = json.load(f)
                pan_offset = data.get('pan_offset', pan_offset)
                tilt_offset = data.get('tilt_offset', tilt_offset)
                # pan_scale and tilt_scale are FORCED to maximum values above
                center_offset_x = data.get('center_offset_x', center_offset_x)
                center_offset_y = data.get('center_offset_y', center_offset_y)
                # spherical_scale is FORCED to maximum value above
                reference_distance = data.get('reference_distance', reference_distance)
                # smoothing_factor is FORCED to 1.0 above
                light_position_x = data.get('light_position_x', light_position_x)
                light_position_y = data.get('light_position_y', light_position_y)
                light_height = data.get('light_height', light_height)
                PAN_CHANNEL = data.get('PAN_CHANNEL', PAN_CHANNEL)
                TILT_CHANNEL = data.get('TILT_CHANNEL', TILT_CHANNEL)
                distance_a1_a2 = data.get('distance_a1_a2', distance_a1_a2)
                anchor_positions = data.get('anchor_positions', anchor_positions)
                pan_range_scale = data.get('pan_range_scale', pan_range_scale)
                tilt_range_scale = data.get('tilt_range_scale', tilt_range_scale)
                pan_range_offset = data.get('pan_range_offset', pan_range_offset)
                tilt_range_offset = data.get('tilt_range_offset', tilt_range_offset)
                pan_min_angle = data.get('pan_min_angle', pan_min_angle)
                pan_max_angle = data.get('pan_max_angle', pan_max_angle)
                tilt_min_angle = data.get('tilt_min_angle', tilt_min_angle)
                tilt_max_angle = data.get('tilt_max_angle', tilt_max_angle)
                # Load named room mappings and active mapping
                global room_mappings, active_room_mapping_name, room_calibration_data, room_calibration_mapping
                room_mappings = data.get('room_mappings', {})
                active_room_mapping_name = data.get('active_room_mapping_name', None)
                room_calibration_data = data.get('room_calibration_data', {})
                room_calibration_mapping = data.get('room_calibration_mapping', None)
            print("Calibration loaded successfully")
        except Exception as e:
            print(f"Failed to load calibration: {e}")

def save_calibration():
    """Save calibration to file"""
    data = {
        'pan_offset': pan_offset,
        'tilt_offset': tilt_offset,
        'pan_scale': pan_scale,
        'tilt_scale': tilt_scale,
        'center_offset_x': center_offset_x,
        'center_offset_y': center_offset_y,
        'spherical_scale': spherical_scale,
        'reference_distance': reference_distance,
        'smoothing_factor': smoothing_factor,
        'light_position_x': light_position_x,
        'light_position_y': light_position_y,
        'light_height': light_height,
        'PAN_CHANNEL': PAN_CHANNEL,
        'TILT_CHANNEL': TILT_CHANNEL,
        'distance_a1_a2': distance_a1_a2,
        'anchor_positions': anchor_positions,
        'pan_range_scale': pan_range_scale,
        'tilt_range_scale': tilt_range_scale,
        'pan_range_offset': pan_range_offset,
        'tilt_range_offset': tilt_range_offset,
        'pan_min_angle': pan_min_angle,
        'pan_max_angle': pan_max_angle,
        'tilt_min_angle': tilt_min_angle,
        'tilt_max_angle': tilt_max_angle,
        'room_mappings': room_mappings,
        'active_room_mapping_name': active_room_mapping_name,
        'room_calibration_data': room_calibration_data,
        'room_calibration_mapping': room_calibration_mapping
    }
    try:
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print("Calibration saved successfully")
    except Exception as e:
        print(f"Failed to save calibration: {e}")

def filter_ranges(ranges):
    """Apply lightweight outlier rejection and optional Kalman filtering to ranges.
    - Clamp sudden jumps (fast outlier suppression) for instant stability
    - If Kalman is enabled, apply it after clamping
    """
    global kalman_filters, kalman_enabled
    
    filtered_ranges = {}
    current_time = time.time()
    
    # Initialize last values store
    if not hasattr(filter_ranges, 'last_values'):
        filter_ranges.last_values = {}
    
    # Conservative jump limit (meters)
    jump_limit = 0.6
    
    for anchor_id, range_val in ranges.items():
        val = float(range_val)
        if val <= 0:
            filtered_ranges[anchor_id] = val
            continue
        
        # Outlier clamp vs last value
        last_val = filter_ranges.last_values.get(anchor_id, val)
        if abs(val - last_val) > jump_limit:
            # Clamp toward last value to suppress spike without adding latency
            if val > last_val:
                val = last_val + jump_limit
            else:
                val = last_val - jump_limit
        
        # Optional Kalman filtering
        if kalman_enabled and anchor_id in kalman_filters:
            val = kalman_filters[anchor_id].update(val, current_time)
        
        filtered_ranges[anchor_id] = val
        filter_ranges.last_values[anchor_id] = val
    
    return filtered_ranges

def filter_position(x, y):
    """Velocity-adaptive position filtering for accuracy without added lag.
    - If moving fast, pass-through for instant response
    - If moving slowly, apply light Kalman smoothing to reduce jitter
    """
    global position_kalman, kalman_enabled, last_x, last_y, last_update_time
    
    if x is None or y is None:
        return x, y
    
    # Compute instantaneous speed
    now_t = time.time()
    dt = max(1e-4, now_t - (last_update_time if 'last_update_time' in globals() else now_t))
    dx = (x - (last_x if 'last_x' in globals() else x))
    dy = (y - (last_y if 'last_y' in globals() else y))
    speed = math.sqrt(dx*dx + dy*dy) / dt
    
    # Threshold for instant vs smoothed (m/s) - ultra-sensitive for instant response
    if not hasattr(filter_position, 'instant_velocity_threshold'):
        filter_position.instant_velocity_threshold = 0.001  # instant response for virtually all movements
    
    # If filter disabled or moving fast, do not smooth
    if not kalman_enabled or speed >= filter_position.instant_velocity_threshold:
        return x, y
    
    current_time = now_t
    
    # Apply Kalman filter to x coordinate
    filtered_x = position_kalman.update(x, current_time)
    
    # Separate filter for y coordinate
    if not hasattr(filter_position, 'y_filter'):
        filter_position.y_filter = LightweightKalmanFilter(
            process_noise=0.001,
            measurement_noise=0.05,
            initial_uncertainty=0.5
        )
    filtered_y = filter_position.y_filter.update(y, current_time)
    
    return filtered_x, filtered_y

def calculate_anchor_distance():
    """Calculate the actual distance between anchors using UWB ranges"""
    global distance_a1_a2
    
    if len(calibration_samples) < 10:
        return None
    
    # Calculate average ranges from calibration samples
    avg_ranges = {"1785": 0.0, "1786": 0.0}
    for sample in calibration_samples:
        for anchor_id, range_val in sample.items():
            avg_ranges[anchor_id] += range_val
    
    for anchor_id in avg_ranges:
        avg_ranges[anchor_id] /= len(calibration_samples)
    
    # Use the minimum range as the anchor distance (when tag is at midpoint)
    min_range = min(avg_ranges.values())
    if min_range > 0:
        # The minimum range should be approximately half the anchor distance
        # Add some correction factor for real-world conditions
        calculated_distance = min_range * 2.1  # Correction factor
        return calculated_distance
    
    return None

def start_auto_calibration():
    """Start automatic calibration process"""
    global auto_calibration_mode, calibration_samples, calibration_step
    auto_calibration_mode = True
    calibration_samples.clear()  # Clear existing samples
    calibration_step = 0
    print("Auto-calibration started. Please move the UWB tag to different positions...")

def stop_auto_calibration():
    """Stop automatic calibration and calculate results"""
    global auto_calibration_mode, distance_a1_a2, anchor_positions
    
    if not auto_calibration_mode:
        return {"success": False, "message": "No calibration in progress"}
    
    auto_calibration_mode = False
    
    if len(calibration_samples) < 10:
        return {"success": False, "message": "Not enough samples collected"}
    
    # Calculate anchor distance
    new_distance = calculate_anchor_distance()
    if new_distance:
        distance_a1_a2 = new_distance
        print(f"Calculated anchor distance: {distance_a1_a2:.3f}m")
    
    # Update anchor positions based on calculated distance
    half_distance = distance_a1_a2 / 2
    anchor_positions["1785"]["x"] = -half_distance
    anchor_positions["1786"]["x"] = half_distance
    
    # Save calibration
    save_calibration()
    
    return {
        "success": True, 
        "message": f"Calibration complete. Anchor distance: {distance_a1_a2:.3f}m",
        "distance": distance_a1_a2,
        "samples": len(calibration_samples)
    }

def collect_calibration_sample():
    """Collect a calibration sample if in auto-calibration mode"""
    global calibration_samples
    
    if not auto_calibration_mode:
        return
    
    # Add current ranges to calibration samples
    if all(r > 0 for r in current_ranges.values()):
        calibration_samples.append(current_ranges.copy())
        print(f"Calibration sample {len(calibration_samples)} collected")

def start_point_learn_calibration():
    """Start point and learn calibration mode"""
    global point_learn_mode, point_learn_samples, point_learn_step
    point_learn_mode = True
    point_learn_samples.clear()  # Clear existing samples
    point_learn_step = 0
    print("Point and learn calibration started!")
    print("Instructions:")
    print("1. Place the UWB tag at a known position")
    print("2. Manually point the spotlight at the tag")
    print("3. Click 'Capture Point' to record the position")
    print("4. Repeat for 3-5 different positions")
    print("5. Click 'Calculate Calibration' when done")

def stop_point_learn_calibration():
    """Stop point and learn calibration and calculate results"""
    global point_learn_mode, manual_mode, last_pan_value, last_tilt_value
    
    try:
        print(f"🔄 Starting calibration calculation with {len(point_learn_samples)} samples")
        
        if not point_learn_mode:
            return {"success": False, "message": "No point and learn calibration in progress"}
        
        if len(point_learn_samples) < 3:
            return {"success": False, "message": "Need at least 3 points for calibration"}
        
        # Calculate calibration from samples
        result = calculate_calibration_from_points()
        
        # Get current UWB position to see where the light should point
        current_x, current_y = calculate_position_improved()
        print(f"📍 Current UWB position after calibration: ({current_x}, {current_y})")
        
        # Calculate where the light should point with new calibration
        if current_x is not None and current_y is not None:
            # Use the new calibration to calculate DMX values
            light_rel_x = current_x - light_position_x
            light_rel_y = current_y - light_position_y
            horiz_dist = math.sqrt(light_rel_x ** 2 + light_rel_y ** 2)

            # Pan angle: positive to the right
            pan_angle = math.degrees(math.atan2(light_rel_x, light_rel_y))

            # Tilt angle: 0° horizontal, + up, - down (fixture-specific; we use positive downwards DMX)
            tilt_angle = -math.degrees(math.atan2(light_height, horiz_dist))

            # Convert angles → DMX using NEW calibration
            pan_raw = ((pan_angle + 180) / 360.0) * 255
            tilt_raw = ((tilt_angle + 90) / 180.0) * 255

            new_pan_value = int(pan_raw * pan_scale + pan_offset)
            new_tilt_value = int(tilt_raw * tilt_scale + tilt_offset)

            # Clamp to DMX range
            new_pan_value = max(0, min(255, new_pan_value))
            new_tilt_value = max(0, min(255, new_tilt_value))
            
            print(f"🎯 New calibration DMX: Pan={new_pan_value}, Tilt={new_tilt_value}")
            print(f"🎯 Previous manual DMX: Pan={current_manual_pan}, Tilt={current_manual_tilt}")
            
            # Update smoothing variables to the new calculated position
            last_pan_value = new_pan_value
            last_tilt_value = new_tilt_value
            
            # Send the new DMX values immediately
            if artnet_socket:
                send_artnet_dmx(new_pan_value, new_tilt_value)
                print(f"🎯 Light moved to new calibrated position: Pan={new_pan_value}, Tilt={new_tilt_value}")
        
        # Now disable calibration mode and enable UWB tracking
        point_learn_mode = False
        manual_mode = False
        auto_follow_mode = True  # Ensure auto-follow is enabled after calibration
        print("✅ Calibration complete - UWB tracking now active with new calibration")
        print(f"[DEBUG] New calibration values: pan_scale={pan_scale}, pan_offset={pan_offset}, tilt_scale={tilt_scale}, tilt_offset={tilt_offset}")
        print(f"[DEBUG] Mode flags: point_learn={point_learn_mode}, manual={manual_mode}, auto_follow={auto_follow_mode}")
        
        return {
            "success": True,
            "message": f"Point and learn calibration complete. {len(point_learn_samples)} points used.",
            "samples": len(point_learn_samples),
            "calibration": result
        }
    except Exception as e:
        print(f"❌ Error in calibration calculation: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Calibration calculation failed: {str(e)}"
        }

def capture_point_learn_sample(data=None):
    """Capture a point and learn sample"""
    global point_learn_samples, current_manual_pan, current_manual_tilt
    
    if not point_learn_mode:
        return {"success": False, "message": "Point and learn mode not active"}
    
    # Use provided coordinates or get current UWB position
    if data and 'uwb_x' in data and 'uwb_y' in data:
        x = float(data['uwb_x'])
        y = float(data['uwb_y'])
        manual_pan = int(data.get('manual_pan', current_manual_pan))
        manual_tilt = int(data.get('manual_tilt', current_manual_tilt))
    else:
        # Get current UWB position
        x, y = calculate_position_improved()
        if x is None or y is None:
            return {"success": False, "message": "No UWB position available"}
        manual_pan = current_manual_pan
        manual_tilt = current_manual_tilt
    
    # Create sample with UWB position and manual DMX values
    sample = {
        "uwb_x": x,
        "uwb_y": y,
        "manual_pan": manual_pan,
        "manual_tilt": manual_tilt,
        "timestamp": time.time()
    }
    
    point_learn_samples.append(sample)
    
    print(f"Point {len(point_learn_samples)} captured:")
    print(f"  UWB Position: ({x:.3f}, {y:.3f})")
    print(f"  Manual DMX: Pan={manual_pan}, Tilt={manual_tilt}")
    
    return {
        "success": True,
        "message": f"Point {len(point_learn_samples)} captured",
        "point_number": len(point_learn_samples),
        "uwb_position": {"x": x, "y": y},
        "manual_dmx": {"pan": manual_pan, "tilt": manual_tilt}
    }

def calculate_calibration_from_points():
    """Calculate calibration parameters from point and learn samples"""
    global pan_offset, tilt_offset, pan_scale, tilt_scale, light_position_x, light_position_y, light_height
    
    if len(point_learn_samples) < 3:
        return {"success": False, "message": "Need at least 3 points"}
    
    print(f"Calculating calibration from {len(point_learn_samples)} samples:")
    for i, sample in enumerate(point_learn_samples):
        print(f"  Sample {i+1}: UWB({sample['uwb_x']:.2f}, {sample['uwb_y']:.2f}) -> DMX({sample['manual_pan']}, {sample['manual_tilt']})")
    
    # Calculate pan calibration using linear regression
    pan_x_values = []
    pan_y_values = []
    tilt_x_values = []
    tilt_y_values = []
    
    for sample in point_learn_samples:
        uwb_x = sample["uwb_x"]
        uwb_y = sample["uwb_y"]
        manual_pan = sample["manual_pan"]
        manual_tilt = sample["manual_tilt"]
        
        # Calculate what the system thinks the DMX should be
        light_relative_x = uwb_x - light_position_x
        light_relative_y = uwb_y - light_position_y
        horizontal_distance = math.sqrt(light_relative_x**2 + light_relative_y**2)
        
        # Calculate pan angle
        pan_angle = math.degrees(math.atan2(light_relative_x, light_relative_y))
        while pan_angle > 180:
            pan_angle -= 360
        while pan_angle < -180:
            pan_angle += 360
        
        # Calculate tilt angle using horizontal distance
        tilt_angle = -math.degrees(math.atan2(light_height, horizontal_distance))

        # Convert to DMX values (matching send_dmx_improved)
        calculated_pan = int(((pan_angle + 180) / 360.0) * 255 + 0.5)
        calculated_tilt = int(((tilt_angle + 90) / 180.0) * 255)
        
        # Store for linear regression
        pan_x_values.append(calculated_pan)
        pan_y_values.append(manual_pan)
        tilt_x_values.append(calculated_tilt)
        tilt_y_values.append(manual_tilt)
        
        print(f"    Calculated: Pan={calculated_pan}, Tilt={calculated_tilt}")
        print(f"    Manual: Pan={manual_pan}, Tilt={manual_tilt}")
        print(f"    Error: Pan={manual_pan - calculated_pan}, Tilt={manual_tilt - calculated_tilt}")
    
    # Calculate linear regression for pan
    print(f"📊 Pan regression data: x_values={pan_x_values}, y_values={pan_y_values}")
    if len(pan_x_values) > 1:
        pan_slope, pan_intercept = linear_regression(pan_x_values, pan_y_values)
        
        # Apply reasonable bounds to prevent extreme calibration values
        # Preserve sign so we can handle inverted pan directions
        if pan_slope >= 0:
            pan_scale = max(0.1, min(5.0, pan_slope))  # Positive scale
        else:
            pan_scale = min(-0.1, max(-5.0, pan_slope))  # Negative scale allowed
        pan_offset = max(-100, min(100, pan_intercept))  # Offset between -100 and 100
        
        print(f"Pan calibration: raw_scale={pan_slope:.3f}, raw_offset={pan_intercept:.3f}")
        print(f"Pan calibration: bounded_scale={pan_scale:.3f}, bounded_offset={pan_offset:.3f}")
        print(f"Global pan_scale updated to: {pan_scale}")
        print(f"Global pan_offset updated to: {pan_offset}")
    else:
        pan_scale = 1.0
        pan_offset = 0.0
        print("Pan calibration: insufficient data, using defaults")
    
    # Calculate linear regression for tilt
    print(f"📊 Tilt regression data: x_values={tilt_x_values}, y_values={tilt_y_values}")
    if len(tilt_x_values) > 1:
        tilt_slope, tilt_intercept = linear_regression(tilt_x_values, tilt_y_values)
        
        # Apply reasonable bounds to prevent extreme calibration values
        # Preserve sign to handle inverted tilt directions
        if tilt_slope >= 0:
            tilt_scale = max(0.1, min(5.0, tilt_slope))
        else:
            tilt_scale = min(-0.1, max(-5.0, tilt_slope))
        tilt_offset = max(-100, min(100, tilt_intercept))  # Offset between -100 and 100
        
        print(f"Tilt calibration: raw_scale={tilt_slope:.3f}, raw_offset={tilt_intercept:.3f}")
        print(f"Tilt calibration: bounded_scale={tilt_scale:.3f}, bounded_offset={tilt_offset:.3f}")
        print(f"Global tilt_scale updated to: {tilt_scale}")
        print(f"Global tilt_offset updated to: {tilt_offset}")
    else:
        tilt_scale = 1.0
        tilt_offset = 0.0
        print("Tilt calibration: insufficient data, using defaults")
    
    # Save calibration
    save_calibration()
    load_calibration()
    
    return {
        "success": True,
        "pan_offset": pan_offset,
        "tilt_offset": tilt_offset,
        "pan_scale": pan_scale,
        "tilt_scale": tilt_scale,
        "samples_used": len(point_learn_samples),
        "pan_samples": pan_x_values,
        "tilt_samples": tilt_x_values
    }

def linear_regression(x_values, y_values):
    """Calculate linear regression slope and intercept"""
    n = len(x_values)
    if n < 2:
        return 1.0, 0.0
    
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_xx = sum(x * x for x in x_values)
    
    # Check for division by zero
    denominator = n * sum_xx - sum_x * sum_x
    if abs(denominator) < 1e-10:  # Very small number, essentially zero
        print(f"⚠️ Linear regression division by zero - all x values are the same: {x_values}")
        return 1.0, 0.0  # Return default values
    
    # Calculate slope and intercept
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    
    print(f"📊 Linear regression: slope={slope:.3f}, intercept={intercept:.3f}")
    return slope, intercept

def calculate_position_improved():
    """Improved position calculation with better accuracy"""
    a1_range = current_ranges["1785"]
    a2_range = current_ranges["1786"]
    
    # Reduced debug output to prevent memory buildup
    if not hasattr(calculate_position_improved, "last_debug_time"):
        calculate_position_improved.last_debug_time = 0
    
    current_time = time.time()
    debug_interval = 1.0  # Only print debug every second
    
    if current_time - calculate_position_improved.last_debug_time > debug_interval:
        print(f"📍 Position calculation - Raw ranges: A1={a1_range:.3f}m, A2={a2_range:.3f}m")
        calculate_position_improved.last_debug_time = current_time
    
    if a1_range <= 0 or a2_range <= 0:
        if current_time - calculate_position_improved.last_debug_time > debug_interval:
            print(f"📍 Position calculation failed - Invalid ranges: A1={a1_range}, A2={a2_range}")
        return None, None
    
    # Apply range filtering
    filtered_ranges = filter_ranges(current_ranges)
    a1_range = filtered_ranges["1785"]
    a2_range = filtered_ranges["1786"]
    
    # Use actual anchor positions for more accurate triangulation
    anchor1 = anchor_positions["1785"]
    anchor2 = anchor_positions["1786"]
    
    # Calculate actual distance between anchors
    dx = anchor2["x"] - anchor1["x"]
    dy = anchor2["y"] - anchor1["y"]
    dz = anchor2["z"] - anchor1["z"]
    actual_distance = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    # Use law of cosines with actual anchor distance, numerically stable
    denom = max(1e-6, 2 * a1_range * actual_distance)
    cos_angle = (a1_range**2 + actual_distance**2 - a2_range**2) / denom
    cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
    
    try:
        angle = math.acos(cos_angle)
        # Stabilized projection
        x = a1_range * math.cos(angle)
        y = max(0.0, a1_range * math.sin(angle))
        
        # Adjust coordinate system to match anchor positions
        x = anchor1["x"] + x
        
        # Apply position filtering
        x, y = filter_position(x, y)
        
        if current_time - calculate_position_improved.last_debug_time > debug_interval:
            print(f"📍 Calculated position: ({x:.3f}, {y:.3f})")
        
        return x, y
    except Exception as e:
        if current_time - calculate_position_improved.last_debug_time > debug_interval:
            print(f"Position calculation error: {e}")
        return None, None

def setup_artnet():
    """Setup Art-Net connection for DMX output"""
    global artnet_socket
    try:
        artnet_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Only set broadcast if not sending to localhost
        if ARTNET_IP != "127.0.0.1":
            artnet_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        print(f"🎨 Art-Net ready for {ARTNET_IP}:{ARTNET_PORT}")
        print(f"🎨 Universe: {ARTNET_UNIVERSE}, Pan Channel: {PAN_CHANNEL}, Tilt Channel: {TILT_CHANNEL}")
        return True
    except Exception as e:
        print(f"❌ Art-Net setup failed: {e}")
        return False

def send_artnet_dmx(pan_value, tilt_value):
    """Send DMX values via Art-Net protocol"""
    if not artnet_socket:
        print("❌ Art-Net socket not initialized")
        return False
    
    try:
        # Build static packet buffer once to minimize allocations and copies
        if not hasattr(send_artnet_dmx, '_packet'):
            header = b'Art-Net\0' + struct.pack('<H', 0x5000) + struct.pack('<H', 14) + b'\x00\x00' + struct.pack('<H', ARTNET_UNIVERSE & 0x7FFF) + struct.pack('<H', 512)
            send_artnet_dmx._packet = bytearray(header + bytes(512))
            send_artnet_dmx._dmx_offset = len(header)
        packet = send_artnet_dmx._packet
        dmx_offset = send_artnet_dmx._dmx_offset
        
        # Set pan and tilt values (convert from 1-based to 0-based indexing)
        packet[dmx_offset + (PAN_CHANNEL - 1)] = pan_value
        packet[dmx_offset + (TILT_CHANNEL - 1)] = tilt_value
        # Optional speed/dimmer channels
        if SPEED_CHANNEL is not None and 1 <= SPEED_CHANNEL <= 512:
            packet[dmx_offset + (SPEED_CHANNEL - 1)] = max(0, min(255, SPEED_VALUE))
        if DIMMER_CHANNEL is not None and 1 <= DIMMER_CHANNEL <= 512:
            packet[dmx_offset + (DIMMER_CHANNEL - 1)] = max(0, min(255, DIMMER_VALUE))
        
        # Send without logging to avoid stalls on stdout (use memoryview to avoid copy)
        artnet_socket.sendto(memoryview(packet), (ARTNET_IP, ARTNET_PORT))
        
        # Record last send time for rate gating
        global last_dmx_send_time
        last_dmx_send_time = time.time()
        
        return True
        
    except Exception as e:
        print(f"❌ Art-Net send error: {e}")
        import traceback
        traceback.print_exc()
        return False

def setup_uwb_connection():
    """Setup UWB data connection with improved buffer management"""
    global uwb_socket, uwb_data
    
    # Clean up any existing connections
    if uwb_data is not None:
        try:
            uwb_data.close()
        except Exception:
            pass
        uwb_data = None
    
    if uwb_socket is not None:
        try:
            uwb_socket.close()
        except Exception:
            pass
        uwb_socket = None
    
    try:
        print("🔌 Setting up UWB connection...")
        # Create new socket with improved buffer settings
        uwb_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        uwb_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Increase buffer sizes to prevent overflow
        uwb_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)  # 64KB receive buffer
        uwb_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)  # 64KB send buffer
        
        # Set socket options for better performance
        uwb_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        uwb_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        
        # Reduce timeout for lower-latency accept/recv
        uwb_socket.settimeout(0.0005)  # Very short timeout for fast connection
        uwb_socket.bind(("0.0.0.0", UDP_PORT))
        uwb_socket.listen(1)
        
        # Set backlog to prevent connection queue overflow
        uwb_socket.listen(5)
        
        print(f"✅ UWB connection ready on port {UDP_PORT}")
        print(f"📡 UWB server listening for connections...")
        print(f"⏳ Waiting for UWB tag to connect...")
        print("⏳ Waiting for UWB tag connection...")
        return True
    except Exception as e:
        print(f"❌ UWB connection setup failed: {e}")
        return False

def read_uwb_data():
    """Read UWB data from the connection with improved buffer management"""
    global uwb_data, current_ranges
    
    # Buffer for accumulating TCP data
    if not hasattr(read_uwb_data, "buffer"):
        read_uwb_data.buffer = ""
    
    # If demo mode is enabled, generate demo data instead
    if demo_mode:
        generate_demo_data()
        return True
    
    if uwb_socket is None:
        return False
    
    # Accept connection if needed
    if uwb_data is None:
        try:
            if uwb_data is not None:
                try:
                    uwb_data.close()
                except Exception:
                    pass
                uwb_data = None
            uwb_data, addr = uwb_socket.accept()
            print(f"✅ Connected to UWB tag at {addr}")
            print(f"🔗 UWB data connection established - ready to receive JSON")
            
            # Set socket options for better performance
            uwb_data.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32768)  # 32KB receive buffer
            uwb_data.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            uwb_data.settimeout(0.0005)  # Very short timeout for fast reads
            
        except socket.timeout:
            return False
        except Exception as e:
            print(f"UWB connection error: {e}")
            return False
    
    # Read data with improved buffer management
    try:
        # Read smaller chunks to prevent buffer overflow
        chunk = uwb_data.recv(512).decode('UTF-8', errors='ignore')
        if chunk:
            read_uwb_data.buffer += chunk
            
            # More aggressive buffer management to prevent overflow
            if len(read_uwb_data.buffer) > 5000:  # Reduced to 5KB limit
                print("⚠️ Buffer overflow, clearing buffer")
                read_uwb_data.buffer = ""
                return False
            
            # Process complete JSON messages
            while '\n' in read_uwb_data.buffer:
                line, read_uwb_data.buffer = read_uwb_data.buffer.split('\n', 1)
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    uwb_json = json.loads(line)
                    
                    # Enhanced debug output to show JSON input processing
                    current_time = time.time()
                    
                    # Always show JSON input for debugging
                    print(f"📡 INPUT JSON: {uwb_json}")
                    
                    # Show processing steps
                    print(f"🔍 Processing JSON with {len(uwb_json.get('links', []))} links")
                    
                    links = uwb_json.get("links", [])
                    
                    for link in links:
                        if "A" in link and "R" in link:
                            incoming_id = str(link["A"])  # ensure string keys
                            mapped_id = ANCHOR_ID_MAP.get(incoming_id, incoming_id)
                            range_val = float(link["R"])
                            current_ranges[mapped_id] = range_val
                            if mapped_id != incoming_id:
                                print(f"📡 Anchor {incoming_id}→{mapped_id}: range = {range_val:.3f}m")
                            else:
                                print(f"📡 Anchor {mapped_id}: range = {range_val:.3f}m")
                    
                    print(f"📡 Updated ranges: {current_ranges}")
                    
                    # Instant path: compute and send DMX immediately on new UWB data
                    try:
                        print(f"🧮 Calculating position from ranges...")
                        pos = calculate_position_improved()
                        if pos and isinstance(pos, tuple):
                            x_i, y_i = pos
                            if x_i is not None and y_i is not None:
                                print(f"📍 Calculated position: X={x_i:.3f}m, Y={y_i:.3f}m")
                                # Update velocity based on delta
                                global last_x, last_y, velocity_x, velocity_y, last_update_time, last_position_time
                                now_t = time.time()
                                dt = max(1e-4, now_t - last_update_time)
                                velocity_x = (x_i - last_x) / dt
                                velocity_y = (y_i - last_y) / dt
                                last_x, last_y = x_i, y_i
                                last_update_time = now_t
                                last_position_time = now_t
                                print(f"⚡ Position updated! Velocity: X={velocity_x:.3f}m/s, Y={velocity_y:.3f}m/s")
                            else:
                                print(f"❌ Invalid position calculated: X={x_i}, Y={y_i}")
                        else:
                            print(f"❌ Position calculation failed: {pos}")
                    except Exception as e:
                        print(f"❌ Error in position calculation: {e}")
                    
                    return True
                    
                except json.JSONDecodeError as e:
                    print(f"❌ JSON Parse Error: {e}")
                    print(f"📄 Raw line: '{line}'")
                    continue
                except Exception as e:
                    print(f"❌ Processing Error: {e}")
                    print(f"📄 Raw line: '{line}'")
                    continue
        else:
            # No UWB connection - this is normal during startup
            return False
            
    except socket.timeout:
        # This is normal - no data received in timeout period
        return False
    except ConnectionResetError:
        print("❌ UWB tag disconnected, waiting for reconnection...")
        uwb_data = None
        return False
    except Exception as e:
        print(f"❌ UWB data read error: {e}")
        uwb_data = None
        return False
    
    return False

def uwb_data_loop():
    """Background thread to continuously read UWB data"""
    print("UWB data thread started")
    while True:
        try:
            read_uwb_data()
            # Collect calibration samples if in auto-calibration mode
            collect_calibration_sample()
        except Exception as e:
            print(f"Exception in UWB data loop: {e}")
        # No delay for instant response

def generate_demo_data():
    """Generate demo position data with improved realism"""
    global demo_time, current_ranges, last_x, last_y, last_position_time
    if not DEMO_ENABLED or not demo_mode:
        return
    
    # Create a more complex motion pattern
    radius = 1.5  # Smaller radius to fit better in room
    center_x = 0.0
    center_y = 1.0
    
    # Add some variation to the motion
    radius_variation = 0.2 * math.sin(demo_time * 0.5)
    current_radius = radius + radius_variation
    
    # Calculate position on circle
    x = center_x + current_radius * math.cos(demo_time)
    y = center_y + current_radius * math.sin(demo_time)
    
    # Ensure position is within reasonable bounds
    x = max(-1.0, min(1.0, x))
    y = max(0.0, min(2.0, y))
    
    # Calculate distances to anchors
    anchor1_x = -distance_a1_a2 / 2
    anchor1_y = 0.0
    anchor2_x = distance_a1_a2 / 2
    anchor2_y = 0.0
    
    # Calculate ranges using distance formula
    range1 = math.sqrt((x - anchor1_x)**2 + (y - anchor1_y)**2)
    range2 = math.sqrt((x - anchor2_x)**2 + (y - anchor2_y)**2)
    
    # Add small realistic noise (±0.02m for better accuracy)
    range1 += random.uniform(-0.02, 0.02)
    range2 += random.uniform(-0.02, 0.02)
    
    # Ensure ranges are valid
    range1 = max(0.1, range1)
    range2 = max(0.1, range2)
    
    # Update ranges
    current_ranges["1785"] = range1
    current_ranges["1786"] = range2
    
    # Store demo position directly for immediate visualization
    last_x = x
    last_y = y
    last_position_time = time.time()
    
    # Increment time for animation
    demo_time += 0.05  # Slightly slower for better visualization

def send_dmx_improved(x, y):
    """Simple auto-follow DMX control - light follows tag automatically"""
    global last_pan_value, last_tilt_value, auto_follow_mode, manual_mode, room_follow_mode_active, room_mapping_autofollow_enabled, room_calibration_mapping
    
    # Reduced debug output to prevent memory buildup
    if not hasattr(send_dmx_improved, "last_debug_time"):
        send_dmx_improved.last_debug_time = 0
    
    current_time = time.time()
    debug_interval = 2.0  # Only print debug every 2 seconds
    
    if current_time - send_dmx_improved.last_debug_time > debug_interval:
        print(f"🎯 send_dmx_improved called: auto_follow={auto_follow_mode}, point_learn={point_learn_mode}")
        send_dmx_improved.last_debug_time = current_time
    
    # Only allow backend auto-follow when enabled and not in calibration or manual
    if not auto_follow_mode or point_learn_mode or manual_mode:
        if current_time - send_dmx_improved.last_debug_time > debug_interval:
            print(f"🚫 send_dmx_improved blocked: auto_follow={auto_follow_mode}, point_learn={point_learn_mode}")
        return None
    
    if x is None or y is None:
        return None
    
    # Predictive lead for tighter following without delay
    # Use current velocity to look slightly ahead, capped to prevent overshoot
    try:
        vx = float(velocity_x)
        vy = float(velocity_y)
    except Exception:
        vx, vy = 0.0, 0.0
    # Lead time in seconds (small to keep response instant)
    lead_time = 0.08
    pred_x = x + vx * lead_time
    pred_y = y + vy * lead_time
    # Cap lead distance (meters)
    dx = pred_x - x
    dy = pred_y - y
    lead_dist = math.hypot(dx, dy)
    max_lead = 0.15
    if lead_dist > 1e-6 and lead_dist > max_lead:
        scale = max_lead / lead_dist
        pred_x = x + dx * scale
        pred_y = y + dy * scale
    # Use predicted position
    x, y = pred_x, pred_y

    # If mapping-based autofollow is enabled and we have a mapping, use it for instant DMX
    if room_mapping_autofollow_enabled and room_calibration_mapping:
        try:
            mapped_pan, mapped_tilt = interpolate_room_position_improved(x, y)
            pan_value = int(max(0, min(255, mapped_pan)))
            tilt_value = int(max(0, min(255, mapped_tilt)))
            final_pan = pan_value
            final_tilt = tilt_value
            last_pan_value = final_pan
            last_tilt_value = final_tilt
            if artnet_socket:
                send_artnet_dmx(final_pan, final_tilt)
            return { 'pan_value': final_pan, 'tilt_value': final_tilt }
        except Exception as e:
            pass

    # Use calibrated geometry (light_position_x/y/height) to compute angles
    light_rel_x = x - light_position_x
    light_rel_y = y - light_position_y
    horiz_dist = math.sqrt(light_rel_x ** 2 + light_rel_y ** 2)

    # Pan angle: positive to the right
    pan_angle = math.degrees(math.atan2(light_rel_x, light_rel_y))

    # Tilt angle calculation based on light orientation
    light_orientation = room_config.get('light_orientation', 'floor')
    tag_height = room_config.get('tag_height', 0.5)  # Get tag height from config
    
    if light_orientation == 'floor':
        # Floor-mounted light (upright) - tilts UP to point at tag
        height_diff = tag_height - light_height  # How much higher the tag is than the light
        if abs(height_diff) < 0.01:  # Tag and light at same height
            tilt_angle = 0.0
        else:
            # Calculate tilt angle: positive = up, negative = down
            tilt_angle = math.degrees(math.atan2(height_diff, horiz_dist))
    else:
        # Ceiling-mounted light (hanging) - tilts DOWN to point at tag
        height_diff = light_height - tag_height  # How much higher the light is than the tag
        if abs(height_diff) < 0.01:  # Tag and light at same height
            tilt_angle = 0.0
        else:
            # Calculate tilt angle: negative = up, positive = down (inverted for ceiling mount)
            tilt_angle = -math.degrees(math.atan2(height_diff, horiz_dist))

    # Convert angles → DMX using current calibration
    # Ensure pan increases to the LEFT (typical stage convention) if needed by allowing negative pan_scale
    pan_raw = ((pan_angle + 180) / 360.0) * 255
    
    # DMX conversion based on light orientation
    if light_orientation == 'floor':
        # Floor-mounted light (upright)
        # 0° = horizontal, +90° = pointing up, -90° = pointing down
        # DMX: 0 = pointing down, 128 = horizontal, 255 = pointing up
        if tilt_angle >= 0:  # Pointing up
            tilt_raw = 128 + (tilt_angle / 90.0) * 127  # 128 to 255
        else:  # Pointing down
            tilt_raw = 128 + (tilt_angle / 90.0) * 128  # 0 to 128
    else:
        # Ceiling-mounted light (hanging) - inverted DMX mapping
        # 0° = horizontal, +90° = pointing down, -90° = pointing up
        # DMX: 0 = pointing up, 128 = horizontal, 255 = pointing down
        if tilt_angle >= 0:  # Pointing down
            tilt_raw = 128 + (tilt_angle / 90.0) * 127  # 128 to 255
        else:  # Pointing up
            tilt_raw = 128 + (tilt_angle / 90.0) * 128  # 0 to 128

    pan_value = int(pan_raw * pan_scale + pan_offset)
    tilt_value = int(tilt_raw * tilt_scale + tilt_offset)

    # Clamp to DMX range
    pan_value = max(0, min(255, pan_value))
    tilt_value = max(0, min(255, tilt_value))
    
    # Instant response - no smoothing
    final_pan = int(pan_value)
    final_tilt = int(tilt_value)
    
    last_pan_value = final_pan
    last_tilt_value = final_tilt
    
    # Send DMX
    if artnet_socket:
        send_artnet_dmx(final_pan, final_tilt)
    
    return {
        'pan_value': final_pan,
        'tilt_value': final_tilt
    }

# Flask routes
@app.route('/')
def index():
    """Serve the main HTML interface"""
    return send_from_directory('.', 'uwb_interface.html')

@app.route('/test_data')
def test_data():
    """Serve the test data HTML file"""
    return send_from_directory('.', 'test_data.html')

@app.route('/network')
def network_info():
    """Serve the network information page"""
    return send_from_directory('.', 'network_info.html')

@app.route('/test')
def test_route():
    """Test route to verify Flask is working"""
    return "Test route working!"

@app.route('/calibration')
def calibration_page():
    """Serve the dedicated calibration interface"""
    return send_from_directory('.', 'integrated_calibration.html')

@app.route('/simple')
def simple_follow_page():
    """Serve the simple auto-follow interface"""
    return send_from_directory('static', 'simple_follow.html')

@app.route('/test_pan_tilt')
def test_pan_tilt():
    """Serve the pan/tilt test page"""
    return send_from_directory('static', 'test_pan_tilt.html')

@app.route('/api/position')
def api_position():
    """Get current position data"""
    global demo_mode, last_x, last_y, last_position_time, manual_mode, room_follow_mode_active
    
    if demo_mode:
        generate_demo_data()
        # In demo mode, use the directly calculated position
        x, y = last_x, last_y
    else:
        x, y = calculate_position_improved()
        # Store position for status tracking
        if x is not None and y is not None:
            last_x = x
            last_y = y
            last_position_time = time.time()
    
    # Do not send DMX here to avoid duplicate updates; DMX is sent on fresh UWB data/background thread
    
    response = {
        "position": {"x": x, "y": y},
        "ranges": current_ranges.copy(),
        "demo_mode": demo_mode
    }
    
    return jsonify(response)

@app.route('/api/calibration', methods=['GET', 'POST'])
def api_calibration():
    """Get or update calibration settings"""
    global pan_offset, tilt_offset, pan_scale, tilt_scale, center_offset_x, center_offset_y
    global spherical_scale, reference_distance, smoothing_factor, light_position_x, light_position_y
    global PAN_CHANNEL, TILT_CHANNEL, light_height, distance_a1_a2, anchor_positions
    global pan_range_scale, tilt_range_scale, pan_range_offset, tilt_range_offset
    global pan_min_angle, pan_max_angle, tilt_min_angle, tilt_max_angle
    
    if request.method == 'POST':
        data = request.get_json()
        if data is None:
            return jsonify({'success': False, 'error': 'No JSON provided'}), 400
        
        # Update calibration values
        pan_offset = data.get('pan_offset', pan_offset)
        tilt_offset = data.get('tilt_offset', tilt_offset)
        pan_scale = data.get('pan_scale', pan_scale)
        tilt_scale = data.get('tilt_scale', tilt_scale)
        center_offset_x = data.get('center_offset_x', center_offset_x)
        center_offset_y = data.get('center_offset_y', center_offset_y)
        spherical_scale = data.get('spherical_scale', spherical_scale)
        reference_distance = data.get('reference_distance', reference_distance)
        smoothing_factor = data.get('smoothing_factor', smoothing_factor)
        light_position_x = data.get('light_position_x', light_position_x)
        light_position_y = data.get('light_position_y', light_position_y)
        light_height = data.get('light_height', light_height)
        PAN_CHANNEL = int(data.get('PAN_CHANNEL', PAN_CHANNEL))
        TILT_CHANNEL = int(data.get('TILT_CHANNEL', TILT_CHANNEL))
        
        # Extended range parameters
        pan_range_scale = float(data.get('pan_range_scale', pan_range_scale))
        tilt_range_scale = float(data.get('tilt_range_scale', tilt_range_scale))
        pan_range_offset = float(data.get('pan_range_offset', pan_range_offset))
        tilt_range_offset = float(data.get('tilt_range_offset', tilt_range_offset))
        pan_min_angle = float(data.get('pan_min_angle', pan_min_angle))
        pan_max_angle = float(data.get('pan_max_angle', pan_max_angle))
        tilt_min_angle = float(data.get('tilt_min_angle', tilt_min_angle))
        tilt_max_angle = float(data.get('tilt_max_angle', tilt_max_angle))
        
        # Handle anchor distance if provided
        if 'distance_a1_a2' in data:
            global distance_a1_a2, anchor_positions
            distance_a1_a2 = float(data['distance_a1_a2'])
            # Update anchor positions
            half_distance = distance_a1_a2 / 2
            anchor_positions["1785"]["x"] = -half_distance
            anchor_positions["1786"]["x"] = half_distance
        
        save_calibration()
        
        return jsonify({'success': True, 'calibration': {
            'pan_offset': pan_offset,
            'tilt_offset': tilt_offset,
            'pan_scale': pan_scale,
            'tilt_scale': tilt_scale,
            'center_offset_x': center_offset_x,
            'center_offset_y': center_offset_y,
            'spherical_scale': spherical_scale,
            'reference_distance': reference_distance,
            'smoothing_factor': smoothing_factor,
            'light_position_x': light_position_x,
            'light_position_y': light_position_y,
            'light_height': light_height,
            'PAN_CHANNEL': PAN_CHANNEL,
            'TILT_CHANNEL': TILT_CHANNEL,
            'distance_a1_a2': distance_a1_a2,
            'anchor_positions': anchor_positions,
            'pan_range_scale': pan_range_scale,
            'tilt_range_scale': tilt_range_scale,
            'pan_range_offset': pan_range_offset,
            'tilt_range_offset': tilt_range_offset,
            'pan_min_angle': pan_min_angle,
            'pan_max_angle': pan_max_angle,
            'tilt_min_angle': tilt_min_angle,
            'tilt_max_angle': tilt_max_angle
        }})
    
    # GET request
    return jsonify({
        'pan_offset': pan_offset,
        'tilt_offset': tilt_offset,
        'pan_scale': pan_scale,
        'tilt_scale': tilt_scale,
        'center_offset_x': center_offset_x,
        'center_offset_y': center_offset_y,
        'spherical_scale': spherical_scale,
        'reference_distance': reference_distance,
        'smoothing_factor': smoothing_factor,
        'light_position_x': light_position_x,
        'light_position_y': light_position_y,
        'light_height': light_height,
        'PAN_CHANNEL': PAN_CHANNEL,
        'TILT_CHANNEL': TILT_CHANNEL,
        'distance_a1_a2': distance_a1_a2,
        'anchor_positions': anchor_positions,
        'pan_range_scale': pan_range_scale,
        'tilt_range_scale': tilt_range_scale,
        'pan_range_offset': pan_range_offset,
        'tilt_range_offset': tilt_range_offset,
        'pan_min_angle': pan_min_angle,
        'pan_max_angle': pan_max_angle,
        'tilt_min_angle': tilt_min_angle,
        'tilt_max_angle': tilt_max_angle
    })

@app.route('/api/calibration/toggle_pan_invert', methods=['POST'])
def toggle_pan_invert():
    """Quickly invert pan direction by flipping pan_scale sign"""
    global pan_scale
    try:
        # Ensure non-zero to avoid a no-op flip
        if abs(pan_scale) < 0.0001:
            pan_scale = 1.0
        pan_scale = -pan_scale
        save_calibration()
        return jsonify({
            'success': True,
            'pan_scale': pan_scale
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/calibration/toggle_tilt_invert', methods=['POST'])
def toggle_tilt_invert():
    """Quickly invert tilt direction by flipping tilt_scale sign"""
    global tilt_scale
    try:
        if abs(tilt_scale) < 0.0001:
            tilt_scale = 1.0
        tilt_scale = -tilt_scale
        save_calibration()
        return jsonify({
            'success': True,
            'tilt_scale': tilt_scale
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/demo_mode', methods=['GET', 'POST'])
def api_demo_mode():
    """Get or toggle demo mode"""
    global demo_mode
    
    if request.method == 'GET':
        return jsonify({'success': True, 'demo_mode': demo_mode if DEMO_ENABLED else False, 'enabled': DEMO_ENABLED})
    else:  # POST
        # Only allow toggle if enabled
        if not DEMO_ENABLED:
            demo_mode = False
            return jsonify({'success': True, 'demo_mode': False, 'enabled': DEMO_ENABLED, 'message': 'Demo mode disabled'})
        demo_mode = not demo_mode
        # Reset demo time when turning on demo mode
        if demo_mode:
            global demo_time
            demo_time = 0.0
        return jsonify({'success': True, 'demo_mode': demo_mode, 'enabled': DEMO_ENABLED})

@app.route('/api/demo_mode/reset', methods=['POST'])
def reset_demo_mode():
    """Force reset demo mode to OFF"""
    global demo_mode, demo_time
    demo_mode = False
    demo_time = 0.0
    return jsonify({'success': True, 'demo_mode': False, 'enabled': DEMO_ENABLED})

@app.route('/api/demo_mode/enable', methods=['POST'])
def set_demo_enabled():
    """Enable/disable demo mode feature entirely"""
    global DEMO_ENABLED, demo_mode
    data = request.get_json() or {}
    enabled = bool(data.get('enabled', False))
    DEMO_ENABLED = enabled
    if not DEMO_ENABLED:
        demo_mode = False
    return jsonify({'success': True, 'enabled': DEMO_ENABLED, 'demo_mode': demo_mode})

@app.route('/api/auto_calibration/start', methods=['POST'])
def start_calibration():
    """Start automatic calibration"""
    start_auto_calibration()
    return jsonify({"success": True, "message": "Auto-calibration started"})

@app.route('/api/auto_calibration/stop', methods=['POST'])
def stop_calibration():
    """Stop automatic calibration and get results"""
    result = stop_auto_calibration()
    return jsonify(result)

@app.route('/api/auto_calibration/status', methods=['GET'])
def get_calibration_status():
    """Get current calibration status"""
    return jsonify({
        "active": auto_calibration_mode,
        "samples_collected": len(calibration_samples),
        "anchor_distance": distance_a1_a2,
        "anchor_positions": anchor_positions
    })

@app.route('/api/calibration/point_learn/start', methods=['POST'])
def start_point_learn():
    """Start point and learn calibration"""
    start_point_learn_calibration()
    return jsonify({"success": True, "message": "Point and learn calibration started"})

@app.route('/api/calibration/point_learn/stop', methods=['POST'])
def stop_point_learn():
    """Stop point and learn calibration and calculate results"""
    try:
        print("🔄 Stop calibration API called")
        result = stop_point_learn_calibration()
        print(f"🔄 Calibration result: {result}")
        
        # Reload calibration after calculating new parameters
        load_calibration()
        
        print(f"🔄 Returning result: {result}")
        return jsonify(result)
    except Exception as e:
        print(f"❌ Error in stop calibration API: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Calibration calculation failed: {str(e)}"
        }), 500

@app.route('/api/calibration/point_learn/capture', methods=['POST'])
def capture_point():
    """Capture a point and learn sample"""
    data = request.get_json()
    print(f"📸 Capture point called with data: {data}")
    result = capture_point_learn_sample(data)
    print(f"📸 Capture point result: {result}")
    return jsonify(result)

@app.route('/api/calibration/point_learn/status', methods=['GET'])
def get_point_learn_status():
    """Get point and learn calibration status"""
    return jsonify({
        "active": point_learn_mode,
        "samples_collected": len(point_learn_samples),
        "current_manual_pan": current_manual_pan,
        "current_manual_tilt": current_manual_tilt
    })

@app.route('/api/calibration/point_learn/set_manual_dmx', methods=['POST'])
def set_manual_dmx():
    """Set manual DMX values for point and learn calibration"""
    global current_manual_pan, current_manual_tilt, last_pan_value, last_tilt_value
    
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"})
    
    if 'pan' in data:
        current_manual_pan = max(0, min(255, int(data['pan'])))
    if 'tilt' in data:
        current_manual_tilt = max(0, min(255, int(data['tilt'])))
    
    # Update smoothing variables to prevent UWB from overriding manual position
    last_pan_value = current_manual_pan
    last_tilt_value = current_manual_tilt
    
    print(f"🎮 Manual DMX set: Pan={current_manual_pan}, Tilt={current_manual_tilt}")
    print(f"🎮 Updated smoothing variables: last_pan={last_pan_value}, last_tilt={last_tilt_value}")
    
    # Send the manual DMX values to the fixture
    if send_artnet_dmx(current_manual_pan, current_manual_tilt):
        return jsonify({
            "success": True,
            "message": f"Manual DMX set: Pan={current_manual_pan}, Tilt={current_manual_tilt}",
            "pan": current_manual_pan,
            "tilt": current_manual_tilt
        })
    else:
        return jsonify({
            "success": False,
            "message": "Failed to send DMX"
        })

@app.route('/api/calibration/point_learn/samples', methods=['GET'])
def get_point_learn_samples():
    """Get all captured point and learn samples"""
    return jsonify({
        "samples": point_learn_samples,
        "count": len(point_learn_samples)
    })

@app.route('/api/calibration/point_learn/clear', methods=['POST'])
def clear_point_learn_samples():
    """Clear all point and learn samples"""
    global point_learn_samples
    point_learn_samples.clear()  # Use clear() for deque
    return jsonify({
        "success": True,
        "message": "Point and learn samples cleared",
        "count": 0
    })

@app.route('/api/test_artnet', methods=['POST'])
def test_artnet_endpoint():
    """Test Art-Net output with specific values"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"})
    
    pan_value = data.get('pan', 128)
    tilt_value = data.get('tilt', 128)
    
    print(f"🧪 Testing Art-Net: Pan={pan_value}, Tilt={tilt_value}")
    
    if send_artnet_dmx(pan_value, tilt_value):
        return jsonify({
            "success": True,
            "message": f"Art-Net test sent: Pan={pan_value}, Tilt={tilt_value}",
            "pan": pan_value,
            "tilt": tilt_value
        })
    else:
        return jsonify({
            "success": False,
            "message": "Failed to send Art-Net test"
        })

# Room Calibration System
room_calibration_data = {}
room_calibration_mapping = None
room_mappings = {}  # Named mappings persisted across restarts: {name: {mapping, points, saved_at}}
active_room_mapping_name = None

@app.route('/api/calibration/calculate_room_mapping', methods=['POST'])
def calculate_room_mapping():
    """Calculate room calibration mapping using bilinear interpolation for 3x3 grid"""
    global room_calibration_data, room_calibration_mapping
    
    data = request.get_json()
    if not data or 'points' not in data:
        return jsonify({"success": False, "message": "No calibration points provided"})
    
    points = data['points']
    if len(points) < 3:
        return jsonify({"success": False, "message": "Need at least 3 calibration points"})
    
    try:
        # Store calibration data with proper structure
        room_calibration_data = {}
        for point in points:
            position = point['position']
            room_calibration_data[position] = {
                'uwb_x': float(point['uwb_x']),
                'uwb_y': float(point['uwb_y']),
                'uwb_z': float(point.get('uwb_z', 0)),
                'pan': int(point['pan']),
                'tilt': int(point['tilt'])
            }
        
        # Calculate grid boundaries and mapping
        grid_info = calculate_grid_boundaries(room_calibration_data)
        
        room_calibration_mapping = {
            'type': 'bilinear_interpolation',
            'points': room_calibration_data,
            'grid_info': grid_info,
            'calculated_at': time.time()
        }
        
        print(f"🎯 Room calibration mapping calculated with {len(points)} points")
        print(f"🎯 Grid boundaries: {grid_info}")
        
        return jsonify({
            "success": True,
            "message": f"Room calibration mapping calculated with {len(points)} points",
            "points_used": len(points),
            "mapping_type": "bilinear_interpolation",
            "grid_info": grid_info
        })
        
    except Exception as e:
        print(f"❌ Error calculating room calibration: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Calibration calculation failed: {str(e)}"
        })

def calculate_grid_boundaries(calibration_data):
    """Calculate the boundaries and spacing of the calibration grid"""
    if not calibration_data:
        return None
    
    # Extract all UWB coordinates
    x_coords = [data['uwb_x'] for data in calibration_data.values()]
    y_coords = [data['uwb_y'] for data in calibration_data.values()]
    
    # Calculate boundaries
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # Calculate grid spacing (assuming 3x3 grid)
    grid_width = max_x - min_x
    grid_height = max_y - min_y
    
    # Calculate cell dimensions
    cell_width = grid_width / 2 if grid_width > 0 else 0.1
    cell_height = grid_height / 2 if grid_height > 0 else 0.1
    
    return {
        'min_x': min_x,
        'max_x': max_x,
        'min_y': min_y,
        'max_y': max_y,
        'grid_width': grid_width,
        'grid_height': grid_height,
        'cell_width': cell_width,
        'cell_height': cell_height,
        'center_x': (min_x + max_x) / 2,
        'center_y': (min_y + max_y) / 2
    }

@app.route('/api/calibration/predict_room_angles', methods=['POST'])
def predict_room_angles():
    """Predict spotlight angles for a given UWB position using room calibration"""
    global room_calibration_mapping
    
    if not room_calibration_mapping:
        return jsonify({"success": False, "message": "No room calibration mapping available"})
    
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No position data provided"})
    
    x = data.get('x')
    y = data.get('y')
    z = data.get('z', 0)
    
    if x is None or y is None:
        return jsonify({"success": False, "message": "Invalid position coordinates"})
    
    try:
        # Use improved interpolation for better accuracy
        predicted_pan, predicted_tilt = interpolate_room_position_improved(x, y)
        
        return jsonify({
            "success": True,
            "pan": predicted_pan,
            "tilt": predicted_tilt,
            "position": {"x": x, "y": y, "z": z}
        })
        
    except Exception as e:
        print(f"❌ Error predicting room angles: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Angle prediction failed: {str(e)}"
        })

def interpolate_room_position_improved(x, y):
    """Improved interpolation using bilinear interpolation and weighted averaging"""
    global room_calibration_data
    
    if not room_calibration_data:
        raise ValueError("No room calibration data available")
    
    if len(room_calibration_data) < 3:
        # Fallback to nearest neighbor for few points
        return interpolate_nearest_neighbor(x, y)
    
    # Try bilinear interpolation first
    try:
        pan, tilt = interpolate_bilinear(x, y)
        if pan is not None and tilt is not None:
            return pan, tilt
    except Exception as e:
        print(f"Bilinear interpolation failed, falling back to weighted average: {e}")
    
    # Fallback to weighted average interpolation
    return interpolate_weighted_average(x, y)

def interpolate_bilinear(x, y):
    """Bilinear interpolation for 3x3 grid"""
    global room_calibration_data
    
    # Find the 4 nearest calibration points forming a rectangle
    nearest_points = find_nearest_rectangle(x, y)
    
    if len(nearest_points) < 4:
        return None, None
    
    # Sort points by distance and take the closest 4
    sorted_points = sorted(nearest_points, key=lambda p: p['distance'])[:4]
    
    # Calculate weighted average based on inverse distance
    total_weight = 0
    weighted_pan = 0
    weighted_tilt = 0
    
    for point in sorted_points:
        weight = 1.0 / (point['distance'] + 0.001)  # Add small value to prevent division by zero
        total_weight += weight
        weighted_pan += point['data']['pan'] * weight
        weighted_tilt += point['data']['tilt'] * weight
    
    if total_weight > 0:
        final_pan = int(weighted_pan / total_weight)
        final_tilt = int(weighted_tilt / total_weight)
        return final_pan, final_tilt
    
    return None, None

def find_nearest_rectangle(x, y):
    """Find calibration points that form a rectangle around the target position"""
    points = []
    
    for position, data in room_calibration_data.items():
        distance = ((x - data['uwb_x']) ** 2 + (y - data['uwb_y']) ** 2) ** 0.5
        points.append({
            'position': position,
            'data': data,
            'distance': distance,
            'x': data['uwb_x'],
            'y': data['uwb_y']
        })
    
    # Sort by distance
    points.sort(key=lambda p: p['distance'])
    
    # Return all points (will be used for weighted average)
    return points

def interpolate_weighted_average(x, y):
    """Weighted average interpolation using inverse distance weighting"""
    global room_calibration_data
    
    total_weight = 0
    weighted_pan = 0
    weighted_tilt = 0
    
    for position, data in room_calibration_data.items():
        # Calculate distance
        distance = ((x - data['uwb_x']) ** 2 + (y - data['uwb_y']) ** 2) ** 0.5
        
        # Use inverse distance weighting (closer points have more influence)
        if distance < 0.001:  # Very close to a calibration point
            return data['pan'], data['tilt']
        
        weight = 1.0 / (distance ** 2)  # Square of inverse distance for better weighting
        total_weight += weight
        weighted_pan += data['pan'] * weight
        weighted_tilt += data['tilt'] * weight
    
    if total_weight > 0:
        final_pan = int(weighted_pan / total_weight)
        final_tilt = int(weighted_tilt / total_weight)
        return final_pan, final_tilt
    
    # Fallback to center position
    return 128, 128

def interpolate_nearest_neighbor(x, y):
    """Simple nearest neighbor interpolation"""
    global room_calibration_data
    
    min_distance = float('inf')
    best_match = None
    
    for position, data in room_calibration_data.items():
        distance = ((x - data['uwb_x']) ** 2 + (y - data['uwb_y']) ** 2) ** 0.5
        
        if distance < min_distance:
            min_distance = distance
            best_match = data
    
    if best_match:
        return best_match['pan'], best_match['tilt']
    else:
        # Fallback to center position
        return 128, 128

def setup_room(width, length, light_height_param, anchor_distance):
    """Setup room configuration for enhanced calibration"""
    global room_config, room_setup_complete, light_position_x, light_position_y, light_height, distance_a1_a2, anchor_positions
    
    room_config = {
        'width': float(width),
        'length': float(length),
        'light_height': float(light_height_param),
        'anchor_distance': float(anchor_distance)
    }
    
    # Update light position based on room setup
    light_position_x = 0.0  # Center of room width
    light_position_y = 0.0  # Center between anchors (not at front of room)
    light_height = room_config['light_height']
    distance_a1_a2 = room_config['anchor_distance']
    
    # Update anchor positions based on room setup
    half_distance = distance_a1_a2 / 2
    anchor_positions["1785"]["x"] = -half_distance
    anchor_positions["1786"]["x"] = half_distance
    
    room_setup_complete = True
    
    # Calculate distance from light to back wall (where anchors are)
    light_to_back_wall_distance = room_config['length']
    
    return {
        "success": True,
        "message": f"Room setup complete: {width}m x {length}m, Light height: {light_height}m, Anchor distance: {anchor_distance}m, Light to back wall: {light_to_back_wall_distance}m"
    }

def get_room_info():
    """Get current room and light configuration"""
    light_to_back_wall_distance = room_config.get('length', 0.0) if room_setup_complete else 0.0
    
    return {
        "room": room_config,
        "light": {
            "position_x": light_position_x,
            "position_y": light_position_y,
            "height": light_height,
            "orientation": 0.0
        },
        "setup_complete": room_setup_complete,
        "distances": {
            "light_to_back_wall": light_to_back_wall_distance,
            "light_to_anchors": light_to_back_wall_distance,  # Same as back wall since anchors are at back
            "room_length": room_config.get('length', 0.0),
            "tag_position": "between_light_and_anchors",
            "typical_tag_distance_from_light": light_to_back_wall_distance * 0.5,  # Tag typically in middle
            "tag_height": room_config.get('tag_height', 0.5)  # Height of UWB tag
        }
    }

def get_recommended_calibration_points():
    """Get recommended calibration points based on room setup"""
    if not room_setup_complete:
        return []
    
    width = room_config['width']
    length = room_config['length']
    
    # Calculate recommended points based on room dimensions
    # Tag is between light and anchors, so we focus on middle and front areas
    points = [
        {
            "name": "Center Front",
            "x": 0.0,
            "y": length * 0.8,  # 80% toward light
            "description": "Center, close to light"
        },
        {
            "name": "Left Front",
            "x": -width/4,
            "y": length * 0.8,
            "description": "Left side, close to light"
        },
        {
            "name": "Right Front",
            "x": width/4,
            "y": length * 0.8,
            "description": "Right side, close to light"
        },
        {
            "name": "Center Middle",
            "x": 0.0,
            "y": length * 0.5,  # Middle of room
            "description": "Center of room"
        },
        {
            "name": "Left Middle",
            "x": -width/4,
            "y": length * 0.5,
            "description": "Left middle area"
        },
        {
            "name": "Right Middle",
            "x": width/4,
            "y": length * 0.5,
            "description": "Right middle area"
        },
        {
            "name": "Center Back",
            "x": 0.0,
            "y": length * 0.2,  # 20% from back wall
            "description": "Center, closer to anchors"
        },
        {
            "name": "Left Back",
            "x": -width/4,
            "y": length * 0.2,
            "description": "Left side, closer to anchors"
        },
        {
            "name": "Right Back",
            "x": width/4,
            "y": length * 0.2,
            "description": "Right side, closer to anchors"
        }
    ]
    
    return points

@app.route('/api/room/setup', methods=['POST'])
def api_setup_room():
    """Setup room configuration"""
    global room_config
    
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"})
    
    width = float(data.get('width', 1.9))
    length = float(data.get('length', 2.5))
    light_height = float(data.get('light_height', 0.0))
    anchor_distance = float(data.get('anchor_distance', 1.9))
    tag_height = float(data.get('tag_height', 0.5))
    light_orientation = data.get('light_orientation', 'floor')  # New parameter
    
    result = setup_room(width, length, light_height, anchor_distance)
    
    # Update additional parameters
    room_config['tag_height'] = tag_height
    room_config['light_orientation'] = light_orientation
    
    return jsonify({
        "success": True,
        "message": f"Room setup complete: {width}m x {length}m, light height: {light_height}m, anchor distance: {anchor_distance}m, tag height: {tag_height}m, orientation: {light_orientation}"
    })

@app.route('/api/room/info', methods=['GET'])
def api_get_room_info():
    """Get room and light information"""
    result = get_room_info()
    print(f"Room info: {result}")  # Debug output
    return jsonify(result)

@app.route('/api/room/recommended_points', methods=['GET'])
def api_get_recommended_points():
    """Get recommended calibration points"""
    return jsonify({
        "points": get_recommended_calibration_points()
    })

@app.route('/api/auto_follow', methods=['POST'])
def toggle_auto_follow():
    """Toggle auto-follow mode"""
    global auto_follow_mode
    data = request.get_json()
    if data and 'enabled' in data:
        auto_follow_mode = bool(data['enabled'])
    else:
        auto_follow_mode = not auto_follow_mode
    
    return jsonify({
        "success": True,
        "auto_follow_enabled": auto_follow_mode,
        "message": f"Auto-follow mode {'enabled' if auto_follow_mode else 'disabled'}"
    })

@app.route('/api/auto_follow/status', methods=['GET'])
def get_auto_follow_status():
    """Get auto-follow status"""
    return jsonify({
        "auto_follow_enabled": auto_follow_mode,
        "manual_mode": manual_mode,
        "point_learn_mode": point_learn_mode
    })

@app.route('/api/reset_tracking', methods=['POST'])
def reset_tracking():
    """Reset system to UWB tracking mode"""
    global point_learn_mode, manual_mode, auto_follow_mode
    
    print("🔄 Resetting system to UWB tracking mode")
    
    # Reset all modes
    point_learn_mode = False
    manual_mode = False
    auto_follow_mode = True
    
    print(f"✅ System reset: point_learn={point_learn_mode}, manual={manual_mode}, auto_follow={auto_follow_mode}")
    
    return jsonify({
        "success": True,
        "message": "System reset to UWB tracking mode",
        "point_learn_mode": point_learn_mode,
        "manual_mode": manual_mode,
        "auto_follow_mode": auto_follow_mode
    })

@app.route('/api/reset_uwb_connection', methods=['POST'])
def reset_uwb_connection():
    """Reset UWB connection"""
    print("🔄 Resetting UWB connection")
    
    # Reset UWB connection
    success = setup_uwb_connection()
    
    if success:
        return jsonify({
            "success": True,
            "message": "UWB connection reset successfully",
            "port": UDP_PORT
        })
    else:
        return jsonify({
            "success": False,
            "message": "Failed to reset UWB connection"
        })

@app.route('/api/calibration/reload', methods=['POST'])
def reload_calibration():
    """Reload calibration from file"""
    global pan_scale, pan_offset, tilt_scale, tilt_offset
    load_calibration()
    print(f"Calibration reloaded - Current values:")
    print(f"  pan_scale: {pan_scale}")
    print(f"  pan_offset: {pan_offset}")
    print(f"  tilt_scale: {tilt_scale}")
    print(f"  tilt_offset: {tilt_offset}")
    return jsonify({
        "success": True,
        "message": "Calibration reloaded",
        "pan_scale": pan_scale,
        "pan_offset": pan_offset,
        "tilt_scale": tilt_scale,
        "tilt_offset": tilt_offset
    })

@app.route('/api/calibration/anchor_distance', methods=['POST'])
def set_anchor_distance():
    """Manually set anchor distance"""
    global distance_a1_a2, anchor_positions
    data = request.get_json()
    if data and 'distance' in data:
        distance_a1_a2 = float(data['distance'])
        # Update anchor positions
        half_distance = distance_a1_a2 / 2
        anchor_positions["1785"]["x"] = -half_distance
        anchor_positions["1786"]["x"] = half_distance
        save_calibration()
        return jsonify({"success": True, "distance": distance_a1_a2})
    return jsonify({"success": False, "message": "Distance parameter required"})

@app.route('/api/calibration/set_values', methods=['POST'])
def set_calibration_values():
    """Manually set calibration values for testing"""
    global pan_scale, pan_offset, tilt_scale, tilt_offset
    data = request.get_json()
    if data:
        if 'pan_scale' in data:
            pan_scale = float(data['pan_scale'])
        if 'pan_offset' in data:
            pan_offset = float(data['pan_offset'])
        if 'tilt_scale' in data:
            tilt_scale = float(data['tilt_scale'])
        if 'tilt_offset' in data:
            tilt_offset = float(data['tilt_offset'])
        
        save_calibration()
        print(f"Calibration values manually set:")
        print(f"  pan_scale: {pan_scale}")
        print(f"  pan_offset: {pan_offset}")
        print(f"  tilt_scale: {tilt_scale}")
        print(f"  tilt_offset: {tilt_offset}")
        
        return jsonify({
            "success": True,
            "message": "Calibration values set",
            "pan_scale": pan_scale,
            "pan_offset": pan_offset,
            "tilt_scale": tilt_scale,
            "tilt_offset": tilt_offset
        })
    return jsonify({"success": False, "message": "No data provided"})

@app.route('/api/kalman/status', methods=['GET'])
def get_kalman_status():
    """Get current Kalman filter status and parameters"""
    global kalman_enabled, kalman_process_noise, kalman_measurement_noise, kalman_initial_uncertainty
    
    # Get filter states for debugging
    range_filter_states = {}
    for anchor_id, filter_obj in kalman_filters.items():
        range_filter_states[anchor_id] = filter_obj.get_state()
    
    position_filter_state = position_kalman.get_state()
    
    return jsonify({
        "enabled": kalman_enabled,
        "process_noise": kalman_process_noise,
        "measurement_noise": kalman_measurement_noise,
        "initial_uncertainty": kalman_initial_uncertainty,
        "range_filters": range_filter_states,
        "position_filter": position_filter_state
    })

@app.route('/api/kalman/configure', methods=['POST'])
def configure_kalman():
    """Configure Kalman filter parameters"""
    global kalman_enabled, kalman_process_noise, kalman_measurement_noise, kalman_initial_uncertainty
    
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"})
    
    # Update parameters if provided
    if 'enabled' in data:
        kalman_enabled = bool(data['enabled'])
    
    if 'process_noise' in data:
        kalman_process_noise = float(data['process_noise'])
        # Update all range filters
        for filter_obj in kalman_filters.values():
            filter_obj.process_noise = kalman_process_noise
        position_kalman.process_noise = kalman_process_noise
    
    if 'measurement_noise' in data:
        kalman_measurement_noise = float(data['measurement_noise'])
        # Update all range filters
        for filter_obj in kalman_filters.values():
            filter_obj.measurement_noise = kalman_measurement_noise
        position_kalman.measurement_noise = kalman_measurement_noise
    
    if 'initial_uncertainty' in data:
        kalman_initial_uncertainty = float(data['initial_uncertainty'])
        # Update all range filters
        for filter_obj in kalman_filters.values():
            filter_obj.initial_uncertainty = kalman_initial_uncertainty
        position_kalman.initial_uncertainty = kalman_initial_uncertainty
    
    print(f"Kalman filter configured:")
    print(f"  Enabled: {kalman_enabled}")
    print(f"  Process noise: {kalman_process_noise}")
    print(f"  Measurement noise: {kalman_measurement_noise}")
    print(f"  Initial uncertainty: {kalman_initial_uncertainty}")
    
    return jsonify({
        "success": True,
        "message": "Kalman filter configured",
        "enabled": kalman_enabled,
        "process_noise": kalman_process_noise,
        "measurement_noise": kalman_measurement_noise,
        "initial_uncertainty": kalman_initial_uncertainty
    })

@app.route('/api/kalman/reset', methods=['POST'])
def reset_kalman():
    """Reset all Kalman filters to initial state"""
    global kalman_filters, position_kalman
    
    # Reset range filters
    for filter_obj in kalman_filters.values():
        filter_obj.reset()
    
    # Reset position filter
    position_kalman.reset()
    
    # Reset y filter if it exists
    if hasattr(filter_position, 'y_filter'):
        filter_position.y_filter.reset()
    
    print("All Kalman filters reset to initial state")
    
    return jsonify({
        "success": True,
        "message": "All Kalman filters reset"
    })

@app.route('/api/kalman/toggle', methods=['POST'])
def toggle_kalman():
    """Toggle Kalman filter on/off"""
    global kalman_enabled
    
    kalman_enabled = not kalman_enabled
    
    status = "enabled" if kalman_enabled else "disabled"
    print(f"Kalman filter {status}")
    
    return jsonify({
        "success": True,
        "enabled": kalman_enabled,
        "message": f"Kalman filter {status}"
    })

@app.route('/api/status')
def get_system_status():
    """Get overall system status"""
    # Get current position
    current_x = last_x if 'last_x' in globals() and last_x is not None else 0
    current_y = last_y if 'last_y' in globals() and last_y is not None else 0
    
    # Get current DMX values
    current_pan = last_pan_value if 'last_pan_value' in globals() and last_pan_value is not None else 128
    current_tilt = last_tilt_value if 'last_tilt_value' in globals() and last_tilt_value is not None else 128
    
    # Check if we have recent position data (within last 5 seconds)
    has_recent_data = False
    if 'last_position_time' in globals():
        import time
        has_recent_data = (time.time() - last_position_time) < 5
    
    return jsonify({
        "uwb_connected": uwb_socket is not None and has_recent_data,
        "dmx_active": artnet_socket is not None,
        "calibration_loaded": True,  # Calibration is always loaded at startup
        "auto_follow_active": auto_follow_mode,
        "demo_mode_active": demo_mode,
        "manual_mode": manual_mode,
        "last_position": {
            "x": current_x,
            "y": current_y
        },
        "last_dmx": {
            "pan": current_pan,
            "tilt": current_tilt
        },
        "has_recent_data": has_recent_data,
        "position_history_size": len(position_history) if 'position_history' in globals() else 0
    })

@app.route('/api/dmx_status')
def get_dmx_status():
    """Get DMX/Art-Net status"""
    return jsonify({
        "artnet_connected": artnet_socket is not None,
        "artnet_ip": ARTNET_IP,
        "artnet_port": ARTNET_PORT,
        "artnet_universe": ARTNET_UNIVERSE,
        "pan_channel": PAN_CHANNEL,
        "tilt_channel": TILT_CHANNEL,
        "last_pan_value": last_pan_value,
        "last_tilt_value": last_tilt_value
    })

@app.route('/api/memory_status')
def get_memory_status():
    """Get memory usage status"""
    import psutil
    import os
    
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return jsonify({
            "memory_usage_mb": memory_info.rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "calibration_samples_count": len(calibration_samples),
            "point_learn_samples_count": len(point_learn_samples),
            "position_history_size": len(position_history),
            "range_history_size": len(range_history),
            "buffer_size": len(getattr(read_uwb_data, "buffer", "")) if hasattr(read_uwb_data, "buffer") else 0
        })
    except ImportError:
        return jsonify({
            "error": "psutil not available",
            "calibration_samples_count": len(calibration_samples),
            "point_learn_samples_count": len(point_learn_samples),
            "position_history_size": len(position_history),
            "range_history_size": len(range_history)
        })

@app.route('/api/network_buffers')
def get_network_buffer_status():
    """Get network buffer status and health"""
    try:
        # Get UWB buffer status
        uwb_buffer_size = len(getattr(read_uwb_data, "buffer", "")) if hasattr(read_uwb_data, "buffer") else 0
        
        # Get socket status
        uwb_connected = uwb_data is not None
        uwb_socket_active = uwb_socket is not None
        
        # Calculate buffer health
        buffer_health = "healthy"
        if uwb_buffer_size > 4000:
            buffer_health = "critical"
        elif uwb_buffer_size > 3000:
            buffer_health = "warning"
        elif uwb_buffer_size > 1000:
            buffer_health = "elevated"
        
        return jsonify({
            "uwb_buffer_size": uwb_buffer_size,
            "uwb_buffer_health": buffer_health,
            "uwb_connected": uwb_connected,
            "uwb_socket_active": uwb_socket_active,
            "artnet_connected": artnet_socket is not None,
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "timestamp": time.time()
        })

@app.route('/api/network_buffers/clear', methods=['POST'])
def clear_network_buffers():
    """Clear network buffers to resolve overflow issues"""
    try:
        # Clear UWB buffer
        if hasattr(read_uwb_data, "buffer"):
            old_size = len(read_uwb_data.buffer)
            read_uwb_data.buffer = ""
            print(f"🧹 Cleared UWB buffer: {old_size} bytes freed")
        
        # Force garbage collection
        gc.collect()
        
        return jsonify({
            "success": True,
            "message": "Network buffers cleared successfully",
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        })

@app.route('/api/memory_cleanup', methods=['POST'])
def cleanup_memory():
    """Clean up memory by clearing buffers and running garbage collection"""
    global calibration_samples, point_learn_samples, position_history, range_history
    
    # Clear all data structures
    calibration_samples.clear()
    point_learn_samples.clear()
    position_history.clear()
    range_history.clear()
    
    # Clear UWB buffer
    if hasattr(read_uwb_data, "buffer"):
        read_uwb_data.buffer = ""
    
    # Run garbage collection
    gc.collect()
    
    return jsonify({
        "success": True,
        "message": "Memory cleanup completed",
        "calibration_samples_count": len(calibration_samples),
        "point_learn_samples_count": len(point_learn_samples),
        "position_history_size": len(position_history),
        "range_history_size": len(range_history)
    })

@app.route('/api/dmx_test', methods=['POST'])
def test_dmx():
    """Test DMX output with specific values"""
    global manual_mode
    
    data = request.get_json()
    
    # Handle manual mode controls first
    if data:
        if data.get('disable_manual', False):
            manual_mode = False
            return jsonify({"success": True, "message": "Manual mode disabled - UWB tracking enabled"})
        
        if data.get('enable_manual', False):
            manual_mode = True
            return jsonify({"success": True, "message": "Manual mode enabled - UWB tracking disabled"})
    
    # Handle DMX commands
    if data and 'pan' in data and 'tilt' in data:
        pan_val = int(data['pan'])
        tilt_val = int(data['tilt'])
        
        # Clamp values to valid range
        pan_val = max(0, min(255, pan_val))
        tilt_val = max(0, min(255, tilt_val))
        
        print(f"🧪 DMX test: sending pan={pan_val}, tilt={tilt_val}")
        
        # Enable manual mode to prevent automatic DMX sending
        manual_mode = True
        
        if send_artnet_dmx(pan_val, tilt_val):
            print(f"✅ DMX test: pan={pan_val}, tilt={tilt_val} sent successfully")
            return jsonify({"success": True, "message": f"DMX test sent: Pan={pan_val}, Tilt={tilt_val}"})
        else:
            print(f"❌ DMX test: pan={pan_val}, tilt={tilt_val} failed to send")
            return jsonify({"success": False, "message": "Failed to send DMX"})
    
    return jsonify({"success": False, "message": "Pan and tilt values required"})



@app.route('/api/dmx_test_tilt', methods=['POST'])
def test_tilt():
    """Test tilt movement specifically"""
    print("🧪 Starting tilt test")
    
    # Test different tilt positions
    tilt_positions = [0, 64, 128, 192, 255, 128]  # Full range test
    
    for i, tilt_val in enumerate(tilt_positions):
        print(f"🧪 Tilt test {i+1}: sending tilt={tilt_val}")
        if send_artnet_dmx(128, tilt_val):  # Keep pan at center
            print(f"✅ Tilt test {i+1}: {tilt_val} sent successfully")
        else:
            print(f"❌ Tilt test {i+1}: {tilt_val} failed to send")
        # No delay - instant position changes
    
    print("🧪 Tilt test completed")
    return jsonify({"success": True, "message": "Tilt test completed"})

@app.route('/api/dmx_test_extended_range', methods=['POST'])
def test_extended_range():
    """Test extended range movement for calibration"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"})
    
    test_type = data.get('type', 'both')  # 'pan', 'tilt', or 'both'
    duration = data.get('duration', 2.0)  # Duration in seconds
    steps = data.get('steps', 10)  # Number of steps
    
    print(f"Starting extended range test: {test_type}, {duration}s, {steps} steps")
    
    if test_type in ['pan', 'both']:
        # Test pan range
        pan_start = 0
        pan_end = 255
        for i in range(steps + 1):
            pan_val = int(pan_start + (pan_end - pan_start) * i / steps)
            if send_artnet_dmx(pan_val, 128):  # Keep tilt at center
                print(f"Pan test step {i+1}/{steps+1}: {pan_val}")
            # No delay - instant movement
    
    if test_type in ['tilt', 'both']:
        # Test tilt range
        tilt_start = 0
        tilt_end = 255
        for i in range(steps + 1):
            tilt_val = int(tilt_start + (tilt_end - tilt_start) * i / steps)
            if send_artnet_dmx(128, tilt_val):  # Keep pan at center
                print(f"Tilt test step {i+1}/{steps+1}: {tilt_val}")
            # No delay - instant movement
    
    # Return to center
    if send_artnet_dmx(128, 128):
        print("Returned to center position")
    
    return jsonify({
        "success": True, 
        "message": f"Extended range test completed: {test_type}",
        "test_type": test_type,
        "duration": duration,
        "steps": steps
    })

@app.route('/api/dmx_test_calibration_pattern', methods=['POST'])
def test_calibration_pattern():
    """Test specific calibration pattern for range verification"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"})
    
    pattern = data.get('pattern', 'grid')  # 'grid', 'circle', 'cross', 'corners'
    speed = data.get('speed', 0.5)  # Seconds per position
    
    print(f"Starting calibration pattern: {pattern}, speed: {speed}s")
    
    if pattern == 'grid':
        # 3x3 grid pattern
        positions = [
            (0, 0), (128, 0), (255, 0),
            (0, 128), (128, 128), (255, 128),
            (0, 255), (128, 255), (255, 255)
        ]
    elif pattern == 'circle':
        # Circular pattern
        import math
        positions = []
        for i in range(8):
            angle = i * math.pi / 4
            x = int(128 + 100 * math.cos(angle))
            y = int(128 + 100 * math.sin(angle))
            x = max(0, min(255, x))
            y = max(0, min(255, y))
            positions.append((x, y))
    elif pattern == 'cross':
        # Cross pattern
        positions = [
            (128, 0), (128, 64), (128, 128), (128, 192), (128, 255),  # Vertical
            (0, 128), (64, 128), (128, 128), (192, 128), (255, 128)   # Horizontal
        ]
    elif pattern == 'corners':
        # Corner test
        positions = [
            (0, 0), (255, 0), (255, 255), (0, 255), (128, 128)
        ]
    else:
        return jsonify({"success": False, "message": f"Unknown pattern: {pattern}"})
    
    for i, (pan_val, tilt_val) in enumerate(positions):
        if send_artnet_dmx(pan_val, tilt_val):
            print(f"Pattern step {i+1}/{len(positions)}: Pan={pan_val}, Tilt={tilt_val}")
        # No delay - instant pattern movement
    
    # Return to center
    if send_artnet_dmx(128, 128):
        print("Returned to center position")
    
    return jsonify({
        "success": True,
        "message": f"Calibration pattern completed: {pattern}",
        "pattern": pattern,
        "positions": len(positions),
        "speed": speed
    })

@app.route('/api/calibration/set_range', methods=['POST'])
def set_range():
    """Set extended range parameters"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"})
    
    global pan_range_scale, tilt_range_scale, pan_range_offset, tilt_range_offset
    global pan_min_angle, pan_max_angle, tilt_min_angle, tilt_max_angle
    
    # Update range parameters
    if 'pan_range_scale' in data:
        pan_range_scale = float(data['pan_range_scale'])
    if 'tilt_range_scale' in data:
        tilt_range_scale = float(data['tilt_range_scale'])
    if 'pan_range_offset' in data:
        pan_range_offset = float(data['pan_range_offset'])
    if 'tilt_range_offset' in data:
        tilt_range_offset = float(data['tilt_range_offset'])
    if 'pan_min_angle' in data:
        pan_min_angle = float(data['pan_min_angle'])
    if 'pan_max_angle' in data:
        pan_max_angle = float(data['pan_max_angle'])
    if 'tilt_min_angle' in data:
        tilt_min_angle = float(data['tilt_min_angle'])
    if 'tilt_max_angle' in data:
        tilt_max_angle = float(data['tilt_max_angle'])
    
    save_calibration()
    
    return jsonify({
        "success": True,
        "message": "Range parameters updated",
        "pan_range_scale": pan_range_scale,
        "tilt_range_scale": tilt_range_scale,
        "pan_range_offset": pan_range_offset,
        "tilt_range_offset": tilt_range_offset,
        "pan_min_angle": pan_min_angle,
        "pan_max_angle": pan_max_angle,
        "tilt_min_angle": tilt_min_angle,
        "tilt_max_angle": tilt_max_angle
    })

@app.route('/api/calibration/reset_range', methods=['POST'])
def reset_range():
    """Reset extended range parameters to defaults"""
    global pan_range_scale, tilt_range_scale, pan_range_offset, tilt_range_offset
    global pan_min_angle, pan_max_angle, tilt_min_angle, tilt_max_angle
    
    pan_range_scale = 1.0
    tilt_range_scale = 1.0
    pan_range_offset = 0.0
    tilt_range_offset = 0.0
    
    pan_min_angle = -180.0
    pan_max_angle = 180.0
    tilt_min_angle = -90.0
    tilt_max_angle = 90.0
    
    save_calibration()
    
    return jsonify({
        "success": True,
        "message": "Range parameters reset to defaults",
        "pan_range_scale": pan_range_scale,
        "tilt_range_scale": tilt_range_scale,
        "pan_range_offset": pan_range_offset,
        "tilt_range_offset": tilt_range_offset,
        "pan_min_angle": pan_min_angle,
        "pan_max_angle": pan_max_angle,
        "tilt_min_angle": tilt_min_angle,
        "tilt_max_angle": tilt_max_angle
    })

@app.route('/api/network_status')
def get_network_status():
    """Get network connection information"""
    import socket
    try:
        # Get the local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"
    
    hostname = socket.gethostname()
    
    # Get all network interfaces
    interfaces = []
    try:
        for interface_name, interface_addresses in socket.getaddrinfo(socket.gethostname(), None):
            if interface_addresses[0] == socket.AF_INET:  # IPv4 only
                ip = interface_addresses[4][0]
                if ip != '127.0.0.1' and ip != local_ip:  # Exclude localhost
                    interfaces.append(ip)
    except:
        pass
    
    return jsonify({
        "local_ip": local_ip,
        "hostname": hostname,
        "port": 5050,
        "all_interfaces": [local_ip] + interfaces,
        "network_urls": [f"http://{ip}:5050" for ip in [local_ip] + interfaces],
        "status": "online"
    })

@app.route('/api/serverinfo')
def get_server_info():
    """Get server information"""
    import socket
    try:
        # Get the local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"
    
    hostname = socket.gethostname()
    return jsonify({
        "ip": local_ip,
        "port": 5050,
        "hostname": hostname,
        "anchor_distance": distance_a1_a2,
        "anchor_positions": anchor_positions,
        "artnet_connected": artnet_socket is not None,
        "network_url": f"http://{local_ip}:5050"
    })

@app.route('/api/instant_mode', methods=['POST'])
def set_instant_mode():
    """Set system to instant mode for maximum speed and sensitivity"""
    global dmx_update_rate, position_update_rate, velocity_filter, position_history, range_history
    
    try:
        data = request.get_json()
        
        # Update performance settings for instant response - force instant rates
        dmx_update_rate = 0.0  # Instant DMX updates - no delay
        position_update_rate = 0.0  # Instant position updates - no delay
        velocity_filter = 0.0  # No filtering for instant response
        
        # Clear position history for instant response
        position_history_size = data.get('position_history_size', 1)
        range_history_size = data.get('range_history_size', 1)
        
        # Resize history queues
        position_history = deque(maxlen=position_history_size)
        range_history = deque(maxlen=range_history_size)
        
        print(f"⚡ INSTANT MODE ACTIVATED:")
        print(f"  DMX Update Rate: {dmx_update_rate}s ({1/dmx_update_rate:.0f}Hz)")
        print(f"  Position Update Rate: {position_update_rate}s ({1/position_update_rate:.0f}Hz)")
        print(f"  Velocity Filter: {velocity_filter}")
        print(f"  Position History Size: {position_history_size}")
        print(f"  Range History Size: {range_history_size}")
        
        return jsonify({
            "success": True,
            "message": "Instant mode activated successfully",
            "settings": {
                "dmx_update_rate": dmx_update_rate,
                "position_update_rate": position_update_rate,
                "velocity_filter": velocity_filter,
                "position_history_size": position_history_size,
                "range_history_size": range_history_size
            }
        })
        
    except Exception as e:
        print(f"❌ Error setting instant mode: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to set instant mode: {str(e)}"
        }), 500

def background_dmx_sender():
    """Background thread to send DMX data with improved performance"""
    last_gc_time = time.time()
    last_buffer_check = time.time()
    
    while True:
        try:
            # Send DMX continuously for responsive tracking
            if not point_learn_mode:
                now_t = time.time()
                dt = now_t - last_update_time
                # Send DMX if we have recent position data - ultra-tight window for instant response
                if dt <= 0.01:  # Use position if less than 10ms old (instant)
                    send_dmx_improved(last_x, last_y)
                elif dt <= 0.05:  # Predict if less than 50ms old (still very fast)
                    px, py = predict_position(dt)
                    send_dmx_improved(px, py)
            
            # Periodic buffer and memory management
            current_time = time.time()
            
            # Check buffer status every 10 seconds
            if current_time - last_buffer_check > 10:
                check_network_buffers()
                last_buffer_check = current_time
            
            # Garbage collection every 30 seconds
            if current_time - last_gc_time > 30:
                gc.collect()
                last_gc_time = current_time
            
            # Ultra-minimal delay for instant tracking
            time.sleep(0.0001)  # 0.1ms delay for maximum responsiveness
            
        except Exception as e:
            print(f"Error in DMX sender thread: {e}")
            time.sleep(0.1)  # Brief pause on error
            continue

def check_network_buffers():
    """Monitor and manage network buffer usage"""
    global uwb_data, uwb_socket
    
    try:
        # Check UWB buffer size
        if hasattr(read_uwb_data, "buffer"):
            buffer_size = len(read_uwb_data.buffer)
            if buffer_size > 3000:  # Warning threshold
                print(f"⚠️ UWB buffer size: {buffer_size} bytes")
                if buffer_size > 4000:  # Critical threshold
                    print("🚨 Clearing UWB buffer to prevent overflow")
                    read_uwb_data.buffer = ""
        
        # Check if UWB connection is still responsive
        if uwb_data is not None:
            try:
                # Send a small ping to check connection
                uwb_data.send(b"ping")
            except Exception:
                print("🔄 UWB connection lost, reconnecting...")
                uwb_data = None
                
    except Exception as e:
        print(f"Buffer check error: {e}")

@app.route('/api/calibration/room/status', methods=['GET'])
def get_room_calibration_status():
    """Get current room calibration status"""
    global room_calibration_data, room_calibration_mapping, room_follow_mode_active
    
    if not room_calibration_mapping:
        return jsonify({
            "active": False,
            "points_captured": 0,
            "mapping_type": None,
            "grid_info": None
        })
    
    return jsonify({
        "active": True,
        "points_captured": len(room_calibration_data),
        "mapping_type": room_calibration_mapping.get('type'),
        "grid_info": room_calibration_mapping.get('grid_info'),
        "calculated_at": room_calibration_mapping.get('calculated_at'),
        "follow_mode_active": room_follow_mode_active
    })

@app.route('/api/calibration/room/clear', methods=['POST'])
def clear_room_calibration():
    """Clear all room calibration data"""
    global room_calibration_data, room_calibration_mapping
    
    room_calibration_data = {}
    room_calibration_mapping = None
    
    return jsonify({
        "success": True,
        "message": "Room calibration data cleared"
    })

@app.route('/api/calibration/room/points', methods=['GET'])
def get_room_calibration_points():
    """Get all captured room calibration points"""
    global room_calibration_data
    
    return jsonify({
        "points": room_calibration_data,
        "count": len(room_calibration_data)
    })

@app.route('/api/calibration/room/mappings', methods=['GET'])
def list_room_mappings():
    """List available named room mappings"""
    names = list(room_mappings.keys())
    return jsonify({
        "success": True,
        "mappings": names,
        "active": active_room_mapping_name
    })

@app.route('/api/calibration/room/save_mapping', methods=['POST'])
def save_room_mapping_named():
    """Save current room mapping with a provided name"""
    global room_mappings, active_room_mapping_name
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({"success": False, "message": "Name is required"}), 400
    if not room_calibration_mapping or not room_calibration_data:
        return jsonify({"success": False, "message": "No mapping to save"}), 400
    room_mappings[name] = {
        'mapping': room_calibration_mapping,
        'points': room_calibration_data,
        'saved_at': time.time()
    }
    active_room_mapping_name = name
    save_calibration()
    return jsonify({"success": True, "message": f"Mapping '{name}' saved", "active": active_room_mapping_name})

@app.route('/api/calibration/room/load_mapping', methods=['POST'])
def load_room_mapping_named():
    """Load a saved room mapping by name"""
    global room_calibration_mapping, room_calibration_data, active_room_mapping_name
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    if name not in room_mappings:
        return jsonify({"success": False, "message": f"Mapping '{name}' not found"}), 404
    pkg = room_mappings[name]
    room_calibration_mapping = pkg.get('mapping')
    room_calibration_data = pkg.get('points', {})
    active_room_mapping_name = name
    save_calibration()
    return jsonify({"success": True, "message": f"Mapping '{name}' loaded", "active": active_room_mapping_name})

@app.route('/api/calibration/room/delete_mapping', methods=['POST'])
def delete_room_mapping_named():
    """Delete a saved room mapping by name"""
    global room_mappings, active_room_mapping_name
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    if name not in room_mappings:
        return jsonify({"success": False, "message": f"Mapping '{name}' not found"}), 404
    del room_mappings[name]
    if active_room_mapping_name == name:
        active_room_mapping_name = None
    save_calibration()
    return jsonify({"success": True, "message": f"Mapping '{name}' deleted", "active": active_room_mapping_name})

@app.route('/api/calibration/room/follow_mode', methods=['POST'])
def toggle_room_follow_mode():
    """Toggle room follow mode on/off"""
    global auto_follow_mode, manual_mode, room_follow_mode_active, room_mapping_autofollow_enabled
    
    data = request.get_json()
    enable = data.get('enable', True) if data else True
    
    if enable:
        # Enable mapping-based backend auto-follow for instant follow
        room_follow_mode_active = False
        room_mapping_autofollow_enabled = True
        manual_mode = False
        auto_follow_mode = True
        return jsonify({
            "success": True,
            "message": "Room follow mode enabled",
            "mode": "room_follow"
        })
    else:
        # Disable mapping-based auto-follow and revert to normal behavior
        room_mapping_autofollow_enabled = False
        room_follow_mode_active = False
        auto_follow_mode = True
        manual_mode = False
        return jsonify({
            "success": True,
            "message": "Room follow mode disabled",
            "mode": "manual"
        })

@app.route('/api/artnet/configure', methods=['POST'])
def configure_artnet():
    """Configure Art-Net connection for different setups"""
    global ARTNET_IP, ARTNET_PORT, ARTNET_UNIVERSE, PAN_CHANNEL, TILT_CHANNEL, SPEED_CHANNEL, SPEED_VALUE, DIMMER_CHANNEL, DIMMER_VALUE
    
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No configuration data provided"})
    
    # Update configuration
    if 'ip' in data:
        ARTNET_IP = data['ip']
    if 'port' in data:
        ARTNET_PORT = int(data['port'])
    if 'universe' in data:
        ARTNET_UNIVERSE = int(data['universe'])
    if 'pan_channel' in data:
        PAN_CHANNEL = int(data['pan_channel'])
    if 'tilt_channel' in data:
        TILT_CHANNEL = int(data['tilt_channel'])
    if 'speed_channel' in data:
        SPEED_CHANNEL = int(data['speed_channel']) if data['speed_channel'] not in (None, "", 0) else None
    if 'speed_value' in data:
        SPEED_VALUE = int(data['speed_value'])
    if 'dimmer_channel' in data:
        DIMMER_CHANNEL = int(data['dimmer_channel']) if data['dimmer_channel'] not in (None, "", 0) else None
    if 'dimmer_value' in data:
        DIMMER_VALUE = int(data['dimmer_value'])
    
    # Reinitialize Art-Net connection
    success = setup_artnet()
    
    if success:
        return jsonify({
            "success": True,
            "message": f"Art-Net configured: {ARTNET_IP}:{ARTNET_PORT}, Universe:{ARTNET_UNIVERSE}, Pan:{PAN_CHANNEL}, Tilt:{TILT_CHANNEL}, Speed:{SPEED_CHANNEL}, Dimmer:{DIMMER_CHANNEL}",
            "config": {
                "ip": ARTNET_IP,
                "port": ARTNET_PORT,
                "universe": ARTNET_UNIVERSE,
                "pan_channel": PAN_CHANNEL,
                "tilt_channel": TILT_CHANNEL,
                "speed_channel": SPEED_CHANNEL,
                "speed_value": SPEED_VALUE,
                "dimmer_channel": DIMMER_CHANNEL,
                "dimmer_value": DIMMER_VALUE
            }
        })
    else:
        return jsonify({
            "success": False,
            "message": "Failed to configure Art-Net connection"
        })

@app.route('/api/artnet/test_connection', methods=['POST'])
def test_artnet_connection():
    """Test Art-Net connection by sending a test pattern"""
    if not artnet_socket:
        return jsonify({"success": False, "message": "Art-Net socket not initialized"})
    
    try:
        # Send test pattern
        test_patterns = [
            (128, 128),  # Center
            (0, 128),    # Left
            (255, 128),  # Right
            (128, 0),    # Up
            (128, 255),  # Down
            (128, 128)   # Back to center
        ]
        
        for i, (pan, tilt) in enumerate(test_patterns):
            if send_artnet_dmx(pan, tilt):
                print(f"✅ Test pattern {i+1}: Pan={pan}, Tilt={tilt}")
            else:
                print(f"❌ Test pattern {i+1}: Pan={pan}, Tilt={tilt} failed")
        
        return jsonify({
            "success": True,
            "message": "Art-Net test pattern sent successfully",
            "patterns_sent": len(test_patterns)
        })
        
    except Exception as e:
        print(f"❌ Art-Net test failed: {e}")
        return jsonify({
            "success": False,
            "message": f"Art-Net test failed: {str(e)}"
        })

@app.route('/api/artnet/status', methods=['GET'])
def get_artnet_status():
    """Get current Art-Net connection status and configuration"""
    return jsonify({
        "connected": artnet_socket is not None,
        "ip": ARTNET_IP,
        "port": ARTNET_PORT,
        "universe": ARTNET_UNIVERSE,
        "pan_channel": PAN_CHANNEL,
        "tilt_channel": TILT_CHANNEL,
        "speed_channel": SPEED_CHANNEL,
        "speed_value": SPEED_VALUE,
        "dimmer_channel": DIMMER_CHANNEL,
        "dimmer_value": DIMMER_VALUE,
        "last_pan": last_pan_value,
        "last_tilt": last_tilt_value
    })

def predict_position(dt=0.0):
    """Predict next position using constant-velocity model."""
    global last_x, last_y, velocity_x, velocity_y
    if dt <= 0:
        return last_x, last_y
    return last_x + velocity_x * dt, last_y + velocity_y * dt

if __name__ == '__main__':
    print("Starting UWB Simple Display...")
    print("=" * 50)
    
    # Load calibration at startup
    load_calibration()
    
    # Setup default room configuration (this will override any saved values)
    print("Setting up default room configuration...")
    setup_room(1.9, 2.5, 0.0, 1.9)
    print("✅ Default room setup complete")
    print(f"📍 Final anchor distance: {distance_a1_a2:.3f}m")
    print(f"📍 Final anchor positions: A1=({anchor_positions['1785']['x']:.3f}, {anchor_positions['1785']['y']:.3f}), A2=({anchor_positions['1786']['x']:.3f}, {anchor_positions['1786']['y']:.3f})")
    
    # Save the corrected configuration
    save_calibration()
    print("💾 Corrected configuration saved to calibration file")
    
    # Setup Art-Net connection
    if not setup_artnet():
        print("⚠ Art-Net setup failed - continuing without DMX output")
        print("   You can still use the web interface for testing")
    
    # Setup UWB connection
    setup_uwb_connection()
    
    # Start UWB data reader thread
    uwb_thread = threading.Thread(target=uwb_data_loop, daemon=True)
    uwb_thread.start()

    # Start DMX sender thread
    dmx_thread = threading.Thread(target=background_dmx_sender, daemon=True)
    dmx_thread.start()
    
    # Get local IP address for network access
    import socket
    try:
        # Get the local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"
    
    print("\n✅ UWB Simple Display started!")
    print(f"🌐 Web interface available at:")
    print(f"   Local: http://localhost:5050")
    print(f"   Network: http://{local_ip}:5050")
    print(f"📊 Calibration interface: http://{local_ip}:5050/calibration")
    print(f"📡 UWB connection listening on port {UDP_PORT}")
    print(f"💡 Art-Net output to {ARTNET_IP}:{ARTNET_PORT}")
    print("Press Ctrl+C to stop")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5050, debug=False) 