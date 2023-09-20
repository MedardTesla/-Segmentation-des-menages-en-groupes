"""
le fichier clustering, nous avons construit un modèle basé sur les caractéristiques à plus forte variance de notre ensemble de données et créé plusieurs visualisations pour communiquer nos résultats. Dans ce fichier,  nous allons combiner tous ces éléments dans une application web dynamique qui permettra aux utilisateurs de choisir leurs propres caractéristiques, de construire un modèle et d'évaluer ses performances par le biais d'une interface utilisateur graphique. En d'autres termes, vous allez créer un outil qui permettra à n'importe qui de construire un modèle sans code.
"""



import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html, Dash
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



def wrangle(filepath):

    """Read SCF data file into ``DataFrame``.
    Returns only credit fearful households whose net worth is less than $2 million.

    Parameters
    ----------
    filepath : str
        Location of CSV file.
    """
    df = pd.read_csv(filepath)
    # create mask
    mask = (df["TURNFEAR"] == 1) & (df["NETWORTH"] < 2e6)
    df = df[mask]
    return df

df = wrangle("data/SCFP2019.csv")

app = Dash(__name__)

app.layout = html.Div(
    
    [
        html.H1(children = "Enquête sur les finances des consommateurs", style={'textAlign':'center'}),
        html.H2("Caractéristiques à forte variance"),
        dcc.Graph(id="bar-chart"),
        dcc.RadioItems(
            options=[
                {"label":"trimmed", "value":True},
                {"label":"not-trimmed","value":False}
            ],
            value=True,
            id="trim-button"
        ),
        html.H2("K-means Clustering"),
        html.H3("Number of Clusters (k)"),
        dcc.Slider(min=1, max=12, step=1, value=2, id="k-slider"),
        html.Div(id="metrics")
    ]
    
)

def get_high_var_features(trimmed=True, return_feat_names=True):

    """Returns the five highest-variance features of ``df``.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    return_feat_names : bool, default=False
        If ``True``, returns feature names as a ``list``. If ``False``
        returns ``Series``, where index is feature names and values are
        variances.
    """
    if trimmed:
        top_five_features = (
            df.apply(trimmed_var).sort_values().tail(5)
        )
    else:
        top_five_features = df.var().sort_values().tail(5)
    if return_feat_names:
        top_five_features = top_five_features.index.to_list()
    
    return top_five_features

@app.callback(
    Output("bar-chart", "figure"), Input("trim-button", "value")
)
def serve_bar_chart(trimmed=True):

    """Returns a horizontal bar chart of five highest-variance features.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.
    """
    top_five_features = get_high_var_features(trimmed=trimmed, return_feat_names=False)
    fig = px.bar(
        x=top_five_features, y=top_five_features.index, orientation="h"
    )
    fig.update_layout(xaxis_title="Variance", yaxis_title="Features")
    return fig

def get_model_metrics(trimmed=True, k=2, return_metrics=False):

    """Build ``KMeans`` model based on five highest-variance features in ``df``.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.

    return_metrics : bool, default=False
        If ``False`` returns ``KMeans`` model. If ``True`` returns ``dict``
        with inertia and silhouette score.

    """
    # get hegh var feature
    features = get_high_var_features(trimmed=trimmed, return_feat_names=True)
    # create features metrics 
    X = df[features]
    # build model
    model = make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=42))
    model.fit(X)
    
    if return_metrics:
        # calculate inertia
        i = model.named_steps["kmeans"].inertia_
        # calculate silhouette
        ss = silhouette_score(X, model.named_steps["kmeans"].labels_)
        # put result in dict
        metrics = {
            "inertia":round(i),
            "silhouette":round(ss, 3)
        }
        # return metrics
        return metrics
    
    return model

@app.callback(
    Output("metrics", "children"),
    Input("trim-button", "value"),
    Input("k-slider", "value")
)
def serve_metrics(trimmed=True, k=2):

    """Returns list of ``H3`` elements containing inertia and silhouette score
    for ``KMeans`` model.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.
    """
    # get metrics
    metrics = get_model_metrics(trimmed=trimmed, k=k, return_metrics=True)
    # add metrics to html elements
    text = [
        html.H3(f"Inertia: {metrics['inertia']}"),
        html.H3(f"Silhoutte: {metrics['silhouette']}")
    ]
    
    return text

def get_pca_labels():

    """
    ``KMeans`` labels.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.
    """
    
    return X_pca

if __name__ == '__main__':
    app.run(debug=True)

