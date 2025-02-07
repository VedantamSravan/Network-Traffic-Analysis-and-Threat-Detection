# Network Traffic Analysis and Threat Detection

## Overview
This project is designed for analyzing network traffic, identifying suspicious activities, and classifying threats using machine learning techniques. The system reads PCAP files, extracts key network features, and uses a Random Forest model to detect suspicious activity based on IP addresses, domains, and JA3 fingerprints.

## Features
- Parses network packets from PCAP files.
- Extracts IP addresses, domain names, and JA3 hashes.
- Matches extracted features against known suspicious indicators.
- Trains a machine learning model to classify suspicious traffic.
- Visualizes suspicious IPs, domains, and JA3 fingerprints.
- Maps suspicious IP locations on a world map.

## Requirements
Ensure the following Python libraries are installed before running the project:

```sh
pip install pandas scapy scikit-learn matplotlib geopandas
```

## File Structure
```
Network-Traffic-Analysis-and-Threat-Detection/
│── main.py                        # Main script
│── suspicious_ips.txt              # List of suspicious IP addresses
│── suspicious_domains.txt          # List of suspicious domains
│── suspicious_ja3.txt              # List of suspicious JA3 fingerprints
│── output_00007_20210112131445.pcap # Sample PCAP file
│── ne_110m_admin_0_countries.shp   # World map shapefile for geopandas
```

## Usage
### Step 1: Load Suspicious Data
The script reads predefined suspicious IPs, domains, and JA3 signatures from text files:
```python
def load_suspicious_data():
    with open('suspicious_ips.txt', 'r') as f:
        suspicious_ips = set(line.strip() for line in f)
    with open('suspicious_domains.txt', 'r') as f:
        suspicious_domains = set(line.strip() for line in f)
    with open('suspicious_ja3.txt', 'r') as f:
        suspicious_ja3 = set(line.strip() for line in f)
    return suspicious_ips, suspicious_domains, suspicious_ja3
```

### Step 2: Read and Process PCAP Files
Extracts network traffic data from PCAP files:
```python
def read_pcap_to_dataframe(filename):
    suspicious_ips, suspicious_domains, suspicious_ja3 = load_suspicious_data()
    df = pd.DataFrame(columns=["ip_src", "ip_dst", "domain", "ja3", "label"])
    with PcapReader(filename) as pcap_reader:
        for packet in pcap_reader:
            df.append(extract_packet_features(packet, suspicious_ips, suspicious_domains, suspicious_ja3))
    return df
```

### Step 3: Train Machine Learning Model
A Random Forest Classifier is used to classify suspicious traffic:
```python
def train_model(df):
    df.fillna("unknown", inplace=True)
    X = pd.get_dummies(df[["ip_src", "ip_dst", "domain", "ja3"]])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    return model
```

### Step 4: Visualize Data
Generate bar charts for suspicious IPs, domains, and JA3 fingerprints:
```python
def visualize_occurrences(df):
    df['ip_src'].value_counts().head(10).plot(kind='bar', title="Top 10 Suspicious IPs")
    plt.show()
```

### Step 5: Plot Suspicious IPs on World Map
Uses GeoPandas to map suspicious IP locations:
```python
def plot_world_map(suspicious_ips):
    world = gpd.read_file('ne_110m_admin_0_countries.shp')
    locations = {ip: (lat, lon) for ip, (lat, lon) in predefined_geolocation.items() if ip in suspicious_ips}
    suspicious_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([v[1] for v in locations.values()],
                                                                 [v[0] for v in locations.values()]))
    world.boundary.plot()
    suspicious_gdf.plot(color='red', markersize=100)
    plt.show()
```

## Running the Project
Execute the script:
```sh
python main.py
```

## Notes
- Ensure that all required dependencies are installed.
- Update `suspicious_ips.txt`, `suspicious_domains.txt`, and `suspicious_ja3.txt` with latest threat intelligence.
- The PCAP file should contain relevant network traffic for accurate analysis.

