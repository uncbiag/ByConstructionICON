import footsteps
import icon_registration as icon
import torch

import train_knee

if __name__ == "__main__":
    input_shape = [1, 1, 130, 155, 130]
    footsteps.initialize()

    dataset = torch.load(
        "/playpen-ssd/tgreer/ICON_brain_preprocessed_data/stripped/brain_train_2xdown_scaled"
    )

    batch_function = lambda: (
        train_knee.make_batch(dataset),
        train_knee.make_batch(dataset),
    )

    loss = train_knee.make_net(input_shape=input_shape)

    net_par = torch.nn.DataParallel(loss).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.0001)

    net_par.train()
    icon.train_batchfunction(net_par, optimizer, batch_function, unwrapped_net=loss)
