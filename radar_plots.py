import os
import torch
import plotly.graph_objects as go


def create_fig(sim, texts):
    fig = go.Figure()
    sim_pos = torch.where(sim > 0, sim, torch.zeros_like(sim))
    sim_neg = torch.where(sim < 0, -sim, torch.zeros_like(sim))
    max_num = torch.max(torch.stack((sim_pos, sim_neg))).item()
    ticks = [i * max_num / 5 for i in range(5)]
    ticks = [round(num, 2) for num in ticks]
    if not os.path.exists("radars"):
        os.mkdir("radars")
    fig.add_trace(go.Scatterpolar(
        r=sim_pos,
        theta=texts,
        fill='toself',
        name='Positive Projection'
    ))
    fig.add_trace(go.Scatterpolar(
        r=sim_neg,
        theta=texts,
        fill='toself',
        name='Negative Projection'
    ))
    fig.update_layout(

        polar=dict(
            radialaxis=dict(
                tickmode='array',
                tickvals=ticks,
            )),
        font=dict(
            size=23,
        ),
        showlegend=False
    )
    return fig


def show_fig(sim, texts):
    fig = create_fig(sim, texts)
    fig.show()


def save_fig(sim, texts, name, path):
    fig = create_fig(sim, texts)
    fig.write_image(path + f"/{name}.pdf")


def get_scores_for_image(image, mean_image_enc, texts, match_model, device):
    text_embbed = match_model.get_texts_projection(texts)
    image_enc = match_model.get_images_projection(image.unsqueeze(0).to(device))
    image_enc -= mean_image_enc
    sim = (image_enc @ text_embbed.t()).squeeze().detach().to("cpu")
    return sim


def plot_radar(image, match_model, principal_phrases, mean_embedding, device):
    similarity_scores = get_scores_for_image(image, mean_embedding, principal_phrases, match_model, device)
    show_fig(similarity_scores, principal_phrases)
