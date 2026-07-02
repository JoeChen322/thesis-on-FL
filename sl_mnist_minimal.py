import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# -----------------------------
# Client-side model
# -----------------------------
class ClientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


# -----------------------------
# Server-side model
# -----------------------------
class ServerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, smashed_data):
        return self.fc2(smashed_data)


client_model = ClientNet()
server_model = ServerNet()

client_optimizer = optim.SGD(client_model.parameters(), lr=0.01)
server_optimizer = optim.SGD(server_model.parameters(), lr=0.01)

criterion = nn.CrossEntropyLoss()
num_rounds = 10


# -----------------------------
# Fake one batch of MNIST-like data
# -----------------------------
x = torch.randn(32, 1, 28, 28)
y = torch.randint(0, 10, (32,))


for round_idx in range(1, num_rounds + 1):
    # -----------------------------
    # Split Learning forward
    # -----------------------------

    # 1. Client forward
    smashed_data = client_model(x)

    # In real SL, this tensor is sent from client to server.
    # Here we detach it to simulate transmission.
    smashed_data_for_server = smashed_data.detach().requires_grad_()

    # 2. Server forward
    output = server_model(smashed_data_for_server)
    loss = criterion(output, y)

    # -----------------------------
    # Split Learning backward
    # -----------------------------

    server_optimizer.zero_grad()
    client_optimizer.zero_grad()

    # 3. Server backward
    loss.backward()

    # 4. Server updates server-side model
    server_optimizer.step()

    # 5. Server sends gradient of smashed_data back to client
    grad_from_server = smashed_data_for_server.grad

    # 6. Client backward using received gradient
    smashed_data.backward(grad_from_server)

    # 7. Client updates client-side model
    client_optimizer.step()

    pred = output.argmax(dim=1)
    accuracy = (pred == y).float().mean().item()

    print(
        f"round {round_idx:02d}/{num_rounds} "
        f"loss: {loss.item():.4f} "
        f"accuracy: {accuracy:.4f}"
    )
