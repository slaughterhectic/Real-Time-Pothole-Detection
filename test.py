import threading
import requests
from ultralytics import YOLO
import cv2
import numpy as np
from confluent_kafka import Producer
from twilio.rest import Client  # Twilio for sending SMS

# Twilio credentials
TWILIO_ACCOUNT_SID = 'AC08d4ec3a3f72891fe812f161eb6f83f4'
TWILIO_AUTH_TOKEN = '2997af45e81c5a710c11c5ed2e992a9e'
TWILIO_PHONE_NUMBER = '+12566078512'
RECIPIENT_PHONE_NUMBER = '+917908631466'  # Your recipient phone number

# Function to get location based on IP address
def get_location():
    try:
        response = requests.get('https://geolocation-db.com/json/')
        data = response.json()
        city = data['city']
        country = data['country_name']
        latitude = data['latitude']
        longitude = data['longitude']
        return city, country, latitude, longitude
    except Exception as e:
        print("Error fetching location:", e)
        return "Unknown", "Unknown", None, None

# Function to continuously update location in the background
def update_location():
    global current_location
    while True:
        current_location = get_location()

# Function to send SMS using Twilio
def send_sms():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        to=RECIPIENT_PHONE_NUMBER,
        from_=TWILIO_PHONE_NUMBER,
        body="Alert Pothole Ahead"
    )

# Load a model (replace "best.pt" with your actual model path)
model = YOLO("best.pt")
class_names = model.names

# Replace "http://your_mobile_ip:8080/video" with the actual URL provided by the IP Webcam app on your mobile device
cap = cv2.VideoCapture("http://10.7.104.188:8080/video")

# Start the location update thread
location_thread = threading.Thread(target=update_location, daemon=True)
location_thread.start()

# Initialize current location
current_location = ("Unknown", "Unknown", None, None)

# Kafka Producer configuration
conf = {
    'bootstrap.servers': 'pkc-419q3.us-east4.gcp.confluent.cloud:9092',
    'security.protocol': 'SASL_SSL',
    'sasl.mechanisms': 'PLAIN',
    'sasl.username': 'OU2ATAMV7N4Y2EUO',
    'sasl.password': '+h0D3DzPLKKS/fgOmDwVcwHyQD4r4C+N2ED569TLohP71iS4mPWtG34pMkxjIwx6'
}

producer = Producer(**conf)
topic = 'pothole2'  # Replace 'coordinates-topic' with your desired topic name

# Initialize a list to store previously detected coordinates
detected_coordinates = []

# Initialize a flag to track if the alert has been sent
alert_sent = False

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (1020, 500))  # Resize for better processing speed (optional)
    h, w, _ = img.shape

    # Perform object detection on the frame
    results = model.predict(img)

    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs

    if masks is not None:
        masks = masks.data.cpu()
        for seg, box in zip(masks.data.cpu().numpy(), boxes):
            seg = cv2.resize(seg, (w, h))
            contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                d = int(box.cls)
                c = class_names[d]
                x, y, x1, y1 = cv2.boundingRect(contour)

                # Get geolocation information
                city, country, latitude, longitude = current_location

                # Check if location data is valid (latitude and longitude are not None)
                if latitude is not None and longitude is not None:
                    # Publish coordinates to Kafka
                    coord_data = f"{city}, {country} ({latitude}, {longitude})"
                    producer.produce(topic, key=c, value=coord_data.encode('utf-8'))

                    # Check if coordinates are already detected
                    if (latitude, longitude) in detected_coordinates:
                        # If duplicate coordinates found and alert hasn't been sent yet, send an SMS alert
                        if not alert_sent:
                            send_sms()
                            alert_sent = True  # Set the flag to True indicating the alert has been sent
                    else:
                        # If new coordinates, add them to the list and reset the alert flag
                        detected_coordinates.append((latitude, longitude))
                        alert_sent = False

                    # Display location text on the frame
                    location_text = f"{city}, {country} ({latitude}, {longitude})"
                    cv2.putText(img, location_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame with detected potholes (if any)
    cv2.imshow('Pothole Detection', img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
