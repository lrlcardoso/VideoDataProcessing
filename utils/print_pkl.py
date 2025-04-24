import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

# Path to your pkl file
pkl_path = r"C:\Users\s4659771\Documents\MyTurn_Project\Data\Processed\P03\Session3_20250219\Video\VR\Camera1\2025-02-19 10-05-17_kinematic_data_filtered.pkl"

# Load the pickle content
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print("Total frames:", len(data))

# Store similarity scores per ID
similarity_by_id = defaultdict(list)

# Loop through all frames and collect similarities
for idx, frame in enumerate(data):#[:450]):
    ids = frame.get('ids')
    sims = frame.get('similarities')
    
    if ids is not None and sims is not None:
        print(f"Frame {idx}:")
        for pid, sim in zip(ids, sims):
            print(f"  → ID: {pid}, Similarity: {sim:.3f}")
            similarity_by_id[pid].append((idx, sim))

# Plot similarities for each ID
plt.figure(figsize=(12, 6))
for pid, values in similarity_by_id.items():
    frames, sims = zip(*values)
    plt.plot(frames, sims, label=f'ID {pid}')

plt.xlabel('Frame Index')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity per Frame for Each ID')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
