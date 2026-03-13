"""The main module of the app.

Contains most of the functions governing the
different app modes.

"""

import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

import NICE_model



def main():
    """The main function of the app.

    Calls the appropriate mode function, depending on the user's choice
    in the sidebar.

    Returns
    -------
    None
    """
    st.title("LESECQ - SIFFERLIN")

    app_mode = st.sidebar.selectbox(
        "Choix du mode de l'application",
        [
            "Présentation",
            "Modèle NICE"
        ],
    )  # , "Show the source code"])
    if app_mode == "Présentation":
        st.write("Bienvenue sur notre application streamlit !")
        st.write("Vous y trouverez notre implémentation d’un modèle de type NICE sur le dataset MNIST.")
        st.write("L’objectif est de reproduire des images « de type MNIST » à partir de bruits issus d’une distribution logistique.")
    # elif app_mode == "Show the source code":
    #     st.code(get_file_content_as_string("./app.py"))
    else:
        nice()





def nice():
    """Application adaptée au modèle NICE que nous avons codé"""
    st.header("Implémentation du modèle NICE sur le dataset MNIST")
    st.write("L'entraînement se lance automatiquement ; tout changement des paramètres entraîne un nouvel entraînement. Si vous souhaitez entraîner de nouveau un modèle avec les mêmes paramètres, appuyez sur le bouton prévu à cet effet.")

    #Déterminer le device utilisé :
    device = NICE_model.get_device()
    st.write(f"Device: `{device}`")

    #Choisir les paramètres du modèle :
    batch_size = st.slider("Batch size", min_value = 32, max_value = 128, value = 64, step = 32)
    hidden_dim = st.slider("Nb neurones (couche additive)", min_value = 600, max_value = 1600, value = 1000, step = 200)
    num_hidden_layers = st.slider("Nb layers (couche additive)", min_value = 1, max_value = 8, value = 5)
    nb_add = st.slider("Nb couches additives", min_value = 1, max_value = 8, value = 4)
    lr = st.select_slider("Learning rate", options = [1e-5, 1e-4, 1e-3], value=1e-4)
    epochs = st.slider("Epochs d'entraînement", min_value = 1, max_value = 100, value = 2)



    path_ckpt, path_history = NICE_model.checkpoint_paths(
        hidden_dim, num_hidden_layers, nb_add, lr, epochs
    )

    if st.button("Effacer le modèle sauvegardé et ré-entraîner"):
        for path in [path_ckpt, path_history]:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    train_loader, valid_loader, test_loader = NICE_model.get_mnist_loaders(batch_size=batch_size)
    model = NICE_model.NICE(in_dim=392,hid_dim=hidden_dim,out_dim=392,num_hid_lay=num_hidden_layers,nb_add=nb_add,).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    history = []
    if os.path.exists(path_ckpt):
        model.load_state_dict(torch.load(path_ckpt, map_location=device))
        if os.path.exists(path_history):
            history = torch.load(path_history)
        st.write("Modèle NICE préexistant téléchargé.")
    else:
        st.write("Aucun modèle sauvegardé avec ces paramètres. Entraînement en cours...")
        progress_slot = st.empty()
        progress_bar = progress_slot.progress(0, text=f"Training epoch 0/{epochs}")

        def on_epoch_end(current_epoch, total_epochs, epoch_loss):
            progress_pct = int(100 * current_epoch / total_epochs)
            progress_bar.progress(
                progress_pct,
                text=(
                    f"Epoch {current_epoch}/{total_epochs} "
                    f"- loss: {epoch_loss:.4f}"
                ),
            )

        history = NICE_model.train_loop(train_loader,model,optimizer,epochs,device,progress_callback=on_epoch_end)
        progress_slot.empty()
        torch.save(model.state_dict(), path_ckpt)
        torch.save(history, path_history)
        st.write("Entraînement fini et modèle enregistré.")


    if history:
        fig_loss = plt.figure()
        plt.plot(np.arange(1, len(history) + 1), history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Courbe d'entraînement du modèle NICE")
        st.pyplot(fig_loss)

    with st.spinner("Calcul de la log likelihood sur l'entraînement, la validation et le test..."):
        train_avg_log_lik = NICE_model.evaluate_log_likelihood(
            model, train_loader, device
        )
        valid_avg_log_lik = NICE_model.evaluate_log_likelihood(
            model, valid_loader, device
        )
        test_avg_log_lik = NICE_model.evaluate_log_likelihood(
            model, test_loader, device
        )

    metrics_df = pd.DataFrame(
        {
            "split": ["train", "valid", "test"],
            "avg_log_likelihood": [train_avg_log_lik, valid_avg_log_lik, test_avg_log_lik],
        }
    ).set_index("split")
    st.write(metrics_df)

    n_gen = st.slider("Nb d'images à générer", 6, 24, 10, 2)
    generated = NICE_model.generate_samples(model, device, n_samples=n_gen).numpy()
    rows = int(np.ceil(n_gen / 5))
    fig_gen, axes = plt.subplots(rows, 5, figsize=(10, 2 * rows))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n_gen:
            ax.imshow(generated[i], cmap="gray", vmin=0, vmax=1)
    fig_gen.suptitle("Images 'de type MNIST' générées par le modèle NICE")
    st.pyplot(fig_gen)


if __name__ == "__main__":
    main()
