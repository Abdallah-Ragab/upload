import cv2
import ultralytics
import time

def benchmark_yolo_video(video_path, start_frame, end_frame):
    """Runs YOLOv8 on a video clip and calculates performance metrics.

    Args:
        video_path (str): Path to the video file.
        start_frame (int): Starting frame for processing.
        end_frame (int): Ending frame for processing.

    Returns:
        tuple: A tuple containing average latency and FPS.
    """

    # Load the YOLOv8 model
    model = ultralytics.YOLO('yolov8n')  # Replace 'yolov8n' with desired model

    # Open the video capture
    cap = cv2.VideoCapture(video_path)

    # Initialize performance metrics
    inference_times = []

    # Process frames within the specified range
    for frame_num in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            break

        # Perform inference
        start_time = time.time()
        results = model(frame)
        end_time = time.time()

        # Record inference time
        inference_time = end_time - start_time
        inference_times.append(inference_time)

        # Visualize results (optional)
        # results.render()  # Displays the image with bounding boxes and labels

        # Save the frame (optional)
        # cv2.imwrite('frame.jpg', results.imgs[0])
        print(inference_time)

    # Calculate performance metrics
    average_latency = sum(inference_times) / len(inference_times)
    fps = 1 / average_latency
    max_latency = max(inference_times)
    min_latency = min(inference_times)
    metrics = {
        'avg_latency': average_latency,
        'max_latency': max_latency,
        'min_latency': min_latency,
        'min_fps' : 1 / max_latency,
        'max_fps': 1 / min_latency,
        'avg_fps': fps,
    }

    return metrics

# Example usage
video_path = 'demo.mp4'
start_frame = 0
end_frame = 100

metrics = benchmark_yolo_video(video_path, start_frame, end_frame)
print(f"Average Latency: {metrics['avg_latency']:.3f} seconds")
print(f"Max Latency: {metrics['max_latency']:.3f} seconds")
print(f"Min Latency: {metrics['min_latency']:.3f} seconds")
print(f"FPS: {metrics['avg_fps']:.2f}")