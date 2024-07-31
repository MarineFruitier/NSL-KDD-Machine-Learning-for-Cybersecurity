import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import StackingClassifier
import xgboost as xgb


# Configuration de la page
st.set_page_config(
    page_title="Cybersécurité avec NSL-KDD",
    page_icon="🔐",
    layout="wide"
)

# Style des titres
st.markdown("""
    <style>
    .title {
        color: #1F4E79;
        font-size: 60px;
        font-weight: bold;
        font-family: 'Arial', sans-serif;
    }
    .subtitle {
        color: #4682B4;
        font-size: 28px;
        font-weight: bold;
        margin-top: 20px;
    }
    .description {
        color: #000000;
        font-size: 20px;
        margin-top: 20px;
    }
    .chart-title {
        color: #1F4E79;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .section-title {
        color: #1F4E79;
        font-size: 40px;
        font-weight: bold;
        margin-top: 40px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


# Introduction
st.markdown("""
# <span class='title'>NSL-KDD Insights: Analyse des Intrusions Cybernétiques 🔐</span>

**Cette application démontre l'utilisation du machine learning pour améliorer la détection des intrusions et la cybersécurité à l'aide du dataset NSL-KDD. Elle inclut une analyse des données via des visualisations et une démonstration de modèles de machine learning appliqués aux données, vous trouverez un mini-tutoriel afin de savoir où surveiller et récupérer vos données cyber en lien avec le dataset et afin de tester vos propres données dans notre modèle.**

## <span class='subtitle'>Contexte</span>
Les données du dataset NSL-KDD proviennent d'une version améliorée du dataset KDD'99, qui est lui-même basé sur les données capturées lors de la compétition DARPA 1998. Cette compétition a été organisée par le MIT Lincoln Laboratory,  une agence du Département de la Défense des États-Unis, et visait à évaluer les systèmes de détection d'intrusion (IDS). Les données de la compétition DARPA 1998 incluent une large variété de scénarios de trafic réseau, comprenant à la fois des connexions normales et diverses attaques simulées.

Le dataset KDD'99 a été dérivé de ces données, mais il présentait plusieurs problèmes comme des enregistrements redondants et des biais en faveur des enregistrements fréquents, ce qui a limité son efficacité pour l'évaluation des IDS. Pour pallier ces limitations, le dataset NSL-KDD a été développé en 2009 par l'Université du Nouveau-Brunswick, apportant des améliorations telles que la suppression des redondances et une meilleure distribution des niveaux de difficulté des enregistrements.

En tant que benchmark, il aide à améliorer la détection des intrusions dans les systèmes modernes et sensibilise aux différents types d'attaques et leurs impacts.

Afin de faciliter son utilisation et d'améliorer la comparabilité des résultats, le NSL-KDD est divisé en 8 sous-ensembles de données, chacun ayant des caractéristiques spécifiques.
Pour cette analyse, nous utilisons spécifiquement les ensembles de données `KDDTrain+.TXT` et `KDDTest+.TXT`. 

- **KDDTrain+.TXT** : Utilisé pour l'entraînement de nos modèles et pour la partie visualisation.
- **KDDTest+.TXT** : Utilisé pour tester nos modèles, ce dataset permet de vérifier la performance des modèles sur des données qui n'ont pas été vues lors de l'entraînement.
""", unsafe_allow_html=True)

# Fonction st cache pour charger des données
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, encoding='ISO-8859-1')

# Afficher les données
def display_data(dataset_name, data):
    st.markdown(f"## Données du dataset {dataset_name}")
    st.write(data)

# Actions lorsque les boutons sont cliqués
if st.button("Afficher KDDTrain+"):
    data_train = load_data('KDDTrain+.txt')
    display_data("KDDTrain+", data_train)

if st.button("Afficher KDDTest+"):
    data_test = load_data('KDDTest+.txt')
    display_data("KDDTest+", data_test)


# Chiffres Clés en Cybersécurité
st.markdown("""
## <span class='subtitle'>Chiffres clés en Cybersécurité</span>
- **2023** : Le coût moyen d'une violation de données a atteint 4,24 millions de dollars.
- **2024** : On estime que les cyberattaques coûtent aux entreprises plus de 10,5 trillions de dollars par an.
- **Économie mondiale** : Les cyberattaques dans leur ensemble coûtent à l'économie mondiale plus de 6 000 milliards de dollars par an, selon une étude de Cybersecurity Ventures en 2021.
- **Temps de récupération** : Selon une enquête de Radware en 2021, il faut en moyenne 23 jours pour se remettre d'une cyberattaque.
- **Attaques les plus fréquentes** : Phishing, Malware, Ransomware.

Bien que le NSL-KDD ne couvre pas spécifiquement ces types d'attaques, il fournit une base solide pour comprendre et améliorer les systèmes de détection d'intrusion, qui sont essentiels pour protéger contre une variété de menaces.
""", unsafe_allow_html=True)

# Types d'attaques dans NSL-KDD
st.markdown("""
## <span class='subtitle'>Types d'attaques représentées dans le NSL-KDD</span>

#### 1. DoS (Denial of Service)
- **Description** : Saturer un service pour le rendre indisponible.
- **Coût global** : Les attaques DoS coûtent aux entreprises des milliards de dollars chaque année. Par exemple, une étude de Netscout en 2022 a estimé que les attaques DoS pourraient coûter jusqu'à 2,3 milliards de dollars par an.
- **Fréquence** : En 2021, Cloudflare a signalé une augmentation de 125% des attaques DoS par rapport à l'année précédente.
- **Exemple notable** : En 2016, l'attaque DDoS sur Dyn a causé des interruptions majeures sur des sites comme Twitter, Reddit et Netflix, affectant des millions d'utilisateurs.

#### 2. Probe
- **Description** : Tentative d'exploration pour découvrir des vulnérabilités.
- **Coût global** : Bien que plus difficiles à chiffrer, les attaques de type probe peuvent mener à des intrusions plus graves. Elles sont souvent les prémisses d'attaques plus destructrices comme les DoS ou les compromissions de données.
- **Fréquence** : Les scans de ports, une forme courante de probe, sont très fréquents. Une étude de Palo Alto Networks en 2022 a révélé que 95% des organisations ont détecté des tentatives de scans de ports au moins une fois par mois.
- **Exemple notable** : En 2020, un scan massif de ports a été détecté ciblant des vulnérabilités dans les systèmes de gestion de base de données comme MySQL et PostgreSQL. Cela a conduit à des attaques de ransomware sur des entreprises telles que Cognizant et Travelex.

#### 3. U2R (User to Root)
- **Description** : L'utilisateur cherche à obtenir des accès root.
- **Coût global** : Les attaques U2R sont particulièrement dangereuses car elles permettent à un attaquant de prendre un contrôle total du système, potentiellement coûtant des millions en termes de pertes de données et de restauration.
- **Fréquence** : Bien que moins fréquentes que les DoS, ces attaques sont critiques en termes d'impact. Une étude de Verizon en 2021 a montré que les attaques U2R représentent environ 2% des attaques globales mais peuvent avoir des conséquences dévastatrices.
- **Exemple notable** : En 2019, l'attaque "Dirty COW" (CVE-2016-5195) a exploité une vulnérabilité dans le noyau Linux, permettant aux attaquants d'obtenir des privilèges root sur les systèmes affectés, compromettant des données telles que les informations d'identification des utilisateurs et les fichiers système.

#### 4. R2L (Remote to Local)
- **Description** : Tentative d'intrusion depuis une machine distante.
- **Coût global** : Les attaques R2L peuvent entraîner des pertes financières importantes. Une enquête de Ponemon Institute en 2022 a estimé que les violations de sécurité coûtent en moyenne 4,24 millions de dollars par incident.
- **Fréquence** : Les attaques R2L sont courantes et constituent une menace persistante pour les organisations. Selon IBM, en 2021, environ 20% des cyberattaques sont de type R2L.
- **Exemple notable** : En 2021, une attaque R2L via une vulnérabilité dans Microsoft Exchange Server a permis aux attaquants d'accéder à des systèmes internes, compromettant des informations sensibles telles que des données de correspondance confidentielle, des informations financières et des données personnelles des employés.
""", unsafe_allow_html=True)


# Titre de la section des graphiques
st.markdown("""
## <span class='section-title'>Dashboard des graphiques : Dashboard : visualisation et analyses des données (KDDTrain)</span>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data_train = pd.read_csv('KDDTrain+.txt', encoding='ISO-8859-1')
    columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
                'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
                'attack', 'level'])
    data_train.columns = columns
    data_train['total_bytes'] = data_train['src_bytes'] + data_train['dst_bytes']
    return data_train

data_train = load_data()

# Fonction pour le mappage des attaques
def map_attack(attack):
    DoS = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
    Probe = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
    U2R = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
    R2L = ['ftp_write', 'guess_passwd', 'httptunnel',  'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'spy',
           'snmpguess', 'warezclient', 'warezmaster', 'xlock', 'xsnoop']
    if attack in DoS:
        return "DoS"
    elif attack in Probe:
        return "Probe"
    elif attack in U2R:
        return "U2R"
    elif attack in R2L:
        return "R2L"
    else:
        return "Normal"

data_train['attack_category'] = data_train['attack'].apply(map_attack)


# Répartition des attaques par classes
st.markdown("<h1 class='chart-title'>Répartition des attaques par classes</h1>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    labels = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
    sizes = [53.5, 36.5, 9.3, 0.8, 0.1]
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
    explode = (0, 0, 0, 0.1, 0.3)  

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(aspect="equal"))

    # Tracé du graphique en forme de donut
    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops=dict(width=0.3, edgecolor='w')
    )

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    plt.setp(autotexts, size=10, weight="bold")
    ax.legend(wedges, labels, title="Attack categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    ax.set_title("Classification des attaques")

    st.pyplot(fig)
with col2:
    st.markdown("<span class='description'>Ce graphique montre la répartition des différentes classes d'attaques présentes dans le dataset NSL-KDD.</span>", unsafe_allow_html=True)
    st.markdown("<span class='description'>**Analyse** : Les connexions normales sont les plus fréquentes, suivies des attaques DoS puis probe. Les classes R2L et U2R sont particulièrement sous représentées malgré leur importance dans le spectre des cyberattaques, celà nous indique de possibles difficultés pour les modèles de machine learning à détecter ces intrusions. Cette répartition aide à évaluer la robustesse des systèmes de détection d'intrusion (IDS) face à des attaques variées.</span>", unsafe_allow_html=True)


# Répartition des types d'attaques
st.markdown("<h1 class='chart-title'>Répartition des attaques par type</h1>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    abnormal_data = data_train[data_train['attack_category'] != 'Normal']

    attack_counts = abnormal_data['attack_category'].value_counts(normalize=True) * 100

    plt.style.use('ggplot')
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    labels = attack_counts.index
    sizes = attack_counts.values
    explode = (0, 0, 0, 0.1)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(aspect="equal"))
    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops=dict(width=0.3, edgecolor='w')
    )

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    plt.setp(autotexts, size=10, weight="bold")
    ax.legend(wedges, labels, title="Attack categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title("Proportion des attaques par type")
    st.pyplot(fig)
with col2:
    st.markdown("<span class='description'>Ce graphique circulaire montre la proportion des différents types d'attaques dans le dataset NSL-KDD.</span>", unsafe_allow_html=True)
    st.markdown("<span class='description'>**Analyse** : Le graphique révèle un fort déséquilibre des types d'attaques comme nous avons pu le voir plus haut, avec les attaques DoS représentant 78.3% des cas, suivies par les attaques Probe à 19.9%. Les attaques R2L et U2R sont nettement moins fréquentes, à 1.7% et 0.1% respectivement, soulignant la nécessité de méthodes de détection capables de gérer ce déséquilibre.</span>", unsafe_allow_html=True)


# Distribution de la Durée
st.markdown("<h1 class='chart-title'>Distribution de la Durée</h1>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    fig_duration = plt.figure(figsize=(12, 6))
    sns.histplot(data_train['duration'], kde=True, color='blue')
    plt.title('Distribution de la Durée')
    plt.xlabel('Durée')
    plt.ylabel('Fréquence')
    st.pyplot(fig_duration)
with col2:
    st.markdown("<span class='description'>Ce graphique montre la distribution de la durée des connexions enregistrées dans le dataset NSL-KDD. L'axe des x représente la durée en secondes, tandis que l'axe des y représente la fréquence de chaque durée.</span>", unsafe_allow_html=True)
    st.markdown("<span class='description'>**Analyse** : La majorité des connexions ont une durée très courte, typique des datasets de trafic réseau. Quelques connexions avec des durées exceptionnellement longues peuvent indiquer des activités suspectes.</span>", unsafe_allow_html=True)

# Distribution des Taux d'erreur de serveur par protocole
st.markdown("<h1 class='chart-title'>Distribution des taux d'erreur de serveur par Protocole</h1>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    fig_error_rate = plt.figure(figsize=(12, 6))
    sns.violinplot(x='protocol_type', y='serror_rate', data=data_train, palette='Set3')
    plt.title('Distribution des taux d\'erreur de serveur par protocole')
    plt.xlabel('Type de protocole')
    plt.ylabel('Taux d\'erreur de serveur')
    st.pyplot(fig_error_rate)
with col2:
    st.markdown("<span class='description'>Ce graphique montre la distribution des taux d'erreur de serveur (serror_rate) par type de protocole utilisé dans les connexions (TCP, ICMP, UDP).</span>", unsafe_allow_html=True)
    st.markdown("<span class='description'>**Analyse** : Les connexions TCP montrent des taux d'erreur variés, ce qui peut refléter des scénarios où les connexions TCP réussissent ou échouent complètement. Les erreurs de serveur élevées dans les connexions TCP peuvent indiquer des tentatives de manipulation de paquets ou des attaques visant à provoquer des erreurs de traitement sur les serveurs, telles que les attaques de type buffer overflow. Les connexions ICMP n'ont pas d'erreurs de serveur, ce qui est cohérent avec leur utilisation principalement pour des diagnostics réseau. Les connexions UDP montrent également des variations dans les taux d'erreur, souvent associées à des attaques DoS, ce qui souligne l'importance de surveiller les erreurs de serveur pour ces protocoles afin de détecter rapidement les activités suspectes.</span>", unsafe_allow_html=True)

# Comptes de services de destination les plus utilisés par type d'attaque
st.markdown("<h1 class='chart-title'>Comptes de services de destination les plus utilisés par type d'attaque</h1>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    fig_services_by_attack, ax = plt.subplots(figsize=(14, 8))
    attack_srv_count = data_train.groupby(['attack', 'dst_host_srv_count']).size().unstack(fill_value=0)
    top_srv_counts = data_train['dst_host_srv_count'].value_counts().index[:10]
    filtered_attack_srv_count = attack_srv_count[top_srv_counts]
    filtered_attack_srv_count.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
    ax.set_title('Comptes de services de destination les plus utilisés par type d\'attaque')
    ax.set_xlabel('Type d\'attaque')
    ax.set_ylabel('Nombre d\'observations')
    st.pyplot(fig_services_by_attack)
with col2:
    st.markdown("<span class='description'>Ce graphique montre les services de destination les plus utilisés pour chaque type d'attaque. Les barres empilées permettent de visualiser la répartition des services parmi différents types d'attaques.</span>", unsafe_allow_html=True)
    st.markdown("<span class='description'>**Analyse** : Le service avec le compte 255 prédomine, notamment pour les connexions normales, indiquant une forte concentration d'activités réseau régulières vers ce service. Les attaques, en revanche, ciblent une gamme plus diversifiée de services, soulignant que les attaquants visent différents services pour maximiser l'impact. Les attaques DoS et Probe montrent une diversité de cibles, tandis que les attaques R2L et U2R sont associées à des comptes de services moins fréquents, indiquant des tentatives plus ciblées. La concentration élevée sur le service 255 pour certaines attaques, comme neptune, souligne des tentatives de saturation spécifiques.</span>", unsafe_allow_html=True)

# Répartition des Flags
st.markdown("<h1 class='chart-title'>Répartition des Flags</h1>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    fig_flags = plt.figure(figsize=(12, 6))
    sns.countplot(x='flag', data=data_train, palette='Set1')
    plt.title('Répartition des Flags')
    plt.xlabel('Flag')
    plt.ylabel('Nombre d\'observations')
    st.pyplot(fig_flags)
with col2:
    st.markdown("<span class='description'>Ce graphique montre la répartition des différents flags dans le dataset NSL-KDD. Les flags sont des indicateurs de l'état des connexions TCP.</span>", unsafe_allow_html=True)
    st.markdown("<span class='description'>**Analyse** : Le flag SF est le plus fréquent, indiquant des connexions réussies. Les flags S0 et REJ sont courants dans les scans de port et les tentatives d'intrusion, respectivement.</span>", unsafe_allow_html=True)


# Distribution des Total Bytes par type d'Attaque
st.markdown("<h1 class='chart-title'>Distribution des Total Bytes par type d'attaque</h1>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    fig_total_bytes = plt.figure(figsize=(12, 6))
    data_train['log_total_bytes'] = np.log1p(data_train['total_bytes'])
    sns.violinplot(x='attack', y='log_total_bytes', data=data_train, scale='width', inner='quartile', palette='Set2')
    plt.title('Distribution des total Bytes par type d\'attaque')
    plt.xlabel('Type d\'attaque')
    plt.ylabel('Log Total Bytes')
    plt.xticks(rotation=90)
    st.pyplot(fig_total_bytes)
with col2:
    st.markdown("<span class='description'>Ce graphique en violon montre la distribution des total bytes (total d'octets transférés) pour chaque type d'attaque, en utilisant une échelle logarithmique pour mieux visualiser les différences.</span>", unsafe_allow_html=True)
    st.markdown("<span class='description'>**Analyse** : Certaines attaques, comme `portsweep` et `warezmaster`, ont une distribution large des total bytes, indiquant des variations significatives dans le volume de données transférées. Cela peut indiquer des tentatives de reconnaissance extensive ou de vol de données. Les attaques `neptune` et `smurf` montrent une distribution plus concentrée, typique des attaques DoS visant à saturer les ressources avec un volume élevé de paquets similaires. La surveillance des variations dans les total bytes peut aider à identifier des comportements anormaux et à distinguer les types d'attaques basées sur les modèles de transfert de données.</span>", unsafe_allow_html=True)

# Matrice de Corrélation
st.markdown("<h1 class='chart-title'>Matrice de Corrélation</h1>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    fig_corr_matrix = plt.figure(figsize=(12, 10))
    numerical_data = data_train.select_dtypes(include=[np.number])
    correlation_matrix = numerical_data.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('Matrice de corrélation')
    st.pyplot(fig_corr_matrix)
st.markdown("<br><br>", unsafe_allow_html=True)
with col2:
    st.markdown("<span class='description'>Cette matrice de corrélation montre les relations entre différentes caractéristiques du dataset NSL-KDD. Les valeurs de corrélation varient de -1 (corrélation négative forte) à 1 (corrélation positive forte), autour de 0 (il n'y pas de corrélation donc on aura tendance à ne pas conserver ces variables).</span>", unsafe_allow_html=True)
    st.markdown("<span class='description'>**Analyse** : Certaines caractéristiques montrent des corrélations fortes, comme `count` et `srv_count`, indiquant qu'elles varient ensemble de manière significative. Les corrélations faibles ou négatives suggèrent une relation inverse ou une indépendance. Des corrélations trop fortes entre certaines variables peuvent perturber le modèle de machine learning en introduisant de la redondance et en augmentant le risque de surajustement. Il est crucial d'étudier ces corrélations pour sélectionner les variables à inclure dans le modèle. Par exemple, combiner ou éliminer des variables hautement corrélées peut simplifier le modèle et améliorer sa performance.</span>", unsafe_allow_html=True)


# Synthèse Finale graph

st.markdown("""
<h2 style='color: #1F4E79;'>Conclusion</h2>

<h3 style='color: #4682B4;'>Analyse du dashboard et implications pour la Cybersécurité</h3>

<span class='description'>
Les graphiques révèlent plusieurs tendances cruciales pour la cybersécurité : 


**La prédominance des attaques DoS**, qui représentent 78.3% des incidents, indique que **les entreprises doivent renforcer leurs défenses contre ces attaques massives** qui saturent les ressources des services. 
En parallèle, les scans de ports (attacks Probe) montrent une diversité dans les services ciblés, ce qui suggère la nécessité de surveiller attentivement les tentatives d'accès aux services variés pour prévenir les intrusions.

La corrélation élevée entre certaines caractéristiques, comme le taux d'erreur de serveur et les attaques DoS, souligne **l'importance de surveiller les erreurs fréquentes** pour détecter rapidement les activités suspectes. 
De plus, *les flags TCP comme S0 et REJ sont indicateurs de tentatives de scans de port* et d'intrusions, respectivement, nécessitant une attention particulière à ces états de connexion pour une détection précoce.

L'analyse des total bytes transférés et des durées de connexion offre des perspectives supplémentaires. 
Les attaques comme `portsweep` et `warezmaster` montrent une large variation de données transférées, soulignant l'importance de surveiller les comportements de transfert de données pour identifier les activités anormales. 
Enfin, **la durée des connexions, souvent plus courte pour les connexions normales**, peut aider à distinguer les activités malveillantes de celles légitimes.

La répartition des services de destination les plus utilisés montre que **certains services sont particulièrement vulnérables et fréquemment ciblés**.
 Une attention particulière doit être portée aux services couramment utilisés comme **HTTP et FZZ**, qui sont souvent exploités par des attaquants.

**Recommandations Clés**:
- **Renforcer les Défenses Contre les DoS** : Mettez en place des systèmes robustes pour prévenir les saturations de ressources et les interruptions de service.
- **Surveillance des Ports** : Surveillez attentivement les tentatives d'accès aux services variés et configurez des règles de pare-feu précises pour prévenir les intrusions.
- **Détection Précoce** : Utilisez les erreurs de serveur et les flags TCP comme indicateurs pour repérer rapidement les comportements suspects.
- **Analyse des Transferts de Données** : Surveillez les volumes de données transférées pour identifier les activités anormales, notamment celles avec de larges variations.
- **Protection des Services Vulnérables** : Renforcez la sécurité des services fréquemment ciblés, comme HTTP et FTP, pour réduire les risques d'exploitation.

En conclusion, ces analyses offrent une compréhension détaillée des comportements des intrusions et des vulnérabilités du réseau et nous donne des pistes pour se prévenir du mieux possible face aux cyber-intrusions.
</span>
""", unsafe_allow_html=True)

# Titre de la section de Machine learning
st.markdown("""
## <span class='section-title'> Machine Learning appliqué au NSL-KDD</span>
""", unsafe_allow_html=True)

st.markdown("""
## <span class='subtitle'>Choix des métriques à observer</span>

Dans le contexte de l'analyse du dataset NSL-KDD pour la détection d'intrusions, il est crucial de prioriser la réduction des faux négatifs (FN) avant celle des faux positifs (FP). Les faux négatifs représentent les intrusions qui ne sont pas détectées et, par conséquent, laissent le système vulnérable aux attaques. Une intrusion non détectée peut entraîner des pertes importantes de données, des compromissions de sécurité et des dommages considérables aux infrastructures.

Ensuite, bien que les faux positifs puissent entraîner des alertes superflues et une charge de travail accrue pour les analystes, leur impact est généralement moins sévère que celui des intrusions non détectées.

C'est pourquoi nous nous concentrons sur des métriques telles que le recall pour minimiser les faux négatifs, tout en surveillant la précision pour gérer les faux positifs. Une attention particulière est également portée aux autres métriques pour assurer un équilibre global entre la sensibilité et la spécificité du modèle.
""", unsafe_allow_html=True)


st.markdown("""
## <span class='subtitle'>Tests et choix du Modèle pour la détection d'intrusion </span>

Pour l'analyse du dataset NSL-KDD, nous avons testé une variété de modèles de machine learning, incluant des modèles supervisés, non supervisés, et des approches de deep learning. La difficulté principale était de lutter contre le surapprentissage 
(la difficulté du modèle à généraliser sur de nouvelles données) probablement due au déséquilibre des échantillons de données pour les types d'attaques.

### <span class='chart-title'>Modèles de Machine Learning Supervisé</span>
Nous avons expérimenté avec plusieurs algorithmes de machine learning supervisé, tels que :
- **Régression Logistique**
- **Arbre de Décision**
- **Random Forest**
- **SVM (Support Vector Machine)**
- **AdaBoost**
- **XGBoost**
- **LightGBM**

Le modèle XGBoost présente les meilleures performance pour les modèles supervisé. Le modèle de stacking avec XGBoost en métat apprenan est également très correct (meilleure précision mais moins bon recall).

Le XGboost s'est démarqué pour ses résultats au niveau du recall.

### <span class='chart-title'>Modèles de Deep Learning</span>
Les architectures de deep learning testées incluent :
- **CNN (Convolutional Neural Network)**
- **RNN (Recurrent Neural Network)**
- **FNN (Feedforward Neural Network)**
- **ANN (Artificial Neural Network)**

Le modèle ANN a montré les meilleures performances parmi les modèles de deep learning.

### <span class='chart-title'>Modèles Non Supervisés</span>
Pour la détection de patterns sans labels nous avons utilisé le - **KNN (K-Nearest Neighbors)**

### <span class='chart-title'>Prétraitement des Données</span>
Le prétraitement des données a inclus plusieurs étapes essentielles :
- **Regroupement des Bytes** : Combinaison des bytes source et destination pour obtenir une métrique de total bytes.
- **Sélection des Features Pertinentes** : Analyse de corrélation pour identifier et réduire la dimensionnalité, utilisation d'un algorithme de selection des meilleures features selon leur score .
- **Équilibrage des Classes** : Utilisation de SMOTE pour équilibrer les classes.

### <span class='chart-title'>Techniques pour Combattre l'Overfitting</span>
Pour améliorer la robustesse des modèles, nous avons utilisé :
- **Régularisation (L2)** : Appliquée dans les modèles de régression et de SVM pour pénaliser les coefficients excessifs.
- **Validation Croisée** : k-fold cross-validation pour évaluer la performance de manière fiable.
- **Ensemble Methods** : Utilisation de Random Forest, AdaBoost et LightGBM pour une meilleure généralisation.


### <span class='chart-title'>Résultats des modèles</span>
""", unsafe_allow_html=True)

# Les résultats des modèles
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'XGBoost', 'LightGBM', 'AdaBoost', 'KNN', 'FNN', 'CNN', 'RNN', 'ANN', 'Stacking Model']

train_precision = [0.969, 0.999983, 0.999436, 0.994663, 0.992518, 0.999897, 0.999941, 0.997815, 0.994651, 0.994609, 0.993457, 0.994189, 0.9997624562028624]
test_precision = [0.910395, 0.961298, 0.966502, 0.953568, 0.923894, 0.967510, 0.961721, 0.918346, 0.967621, 0.965690, 0.921890, 0.951030, 0.9613342566943675]

train_recall = [0.952908, 0.999898, 0.997356, 0.993511, 0.996748, 1.0, 0.999955, 0.997083, 0.989459, 0.994303, 0.984035, 0.992154, 0.9999703008523655]
test_recall = [0.558993, 0.634897, 0.595854, 0.611362, 0.670745, 0.608106, 0.601075, 0.621415, 0.605517, 0.631702, 0.586814, 0.647756, 0.6490804239401496]

# Création du graphique
col1, col2 = st.columns([1, 2])
with col1:
    fig, ax = plt.subplots(figsize=(14, 8))

    # Précision
    ax.plot(models, train_precision, marker='o', linestyle='-', color='blue', label='Train Precision')
    ax.plot(models, test_precision, marker='o', linestyle='-', color='green', label='Test Precision')

    # Rappel
    ax.plot(models, train_recall, marker='o', linestyle='-', color='pink', label='Train Recall')
    ax.plot(models, test_recall, marker='o', linestyle='-', color='red', label='Test Recall')

    ax.set_xlabel('Modèles')
    ax.set_ylabel('Score')
    ax.set_title('Comparaison des Précisions et Recalls des modèles')
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    st.pyplot(fig)

# Choix du modèle final
st.markdown("""
## <span class='subtitle'>Choix du modèle final : Modèle XGBoost</span>

Après évaluation, nous avons opté pour une approche de **XGBoost** en utilisant la méthode SMOTE. Ce modèle offre une prédiction équilibré et le meilleur recall, notre choix ce porte donc vers lui pour éviter au maximum les faux négatifs lors de l'adaptation du modèle à des données réelles.


**Vous pouvez tester le modèle dans la partie "Simulation", les features avec des données non saisies seront automatiquement remplies par la valeur médiane observée dans le Data_train.**

**Si vous ne savez pas où trouver vos données de cybersécurité vous pouvez suivre le mini-tutoriel ci dessous en cliquant dessus. Il a été crée afin d'observer de comprendre et de collecter les données en liens avec les 12 features les plus importantes du NSL-KDD.**
""", unsafe_allow_html=True)

# Section de tutoriel cachée 
with st.expander("Mini Tutoriel : Accéder aux Logs réseau et entrées pour voir ses information de cybersécurité"):
    st.markdown("""
    ### Qu'est-ce qu'un log réseau ?
    Un log réseau est un fichier ou une base de données qui enregistre les activités du réseau. Cela inclut des informations sur les connexions établies, les tentatives de connexion échouées, les volumes de données transférées, etc. Ces logs sont généralement générés par des dispositifs réseau tels que les routeurs, les pare-feu, et les serveurs.
    
    ### Où trouver les logs ?
    Les logs réseau sont souvent stockés sur les dispositifs réseau eux-mêmes ou sur des serveurs de log centralisés. Voici comment accéder aux logs sur des dispositifs courants :
    
    **Routeurs et Pare-feu :**
    
    **Accès via l'interface web :**
    
    1. Ouvrez votre navigateur web.
    2. Entrez l'adresse IP du routeur ou du pare-feu (souvent quelque chose comme 192.168.1.1 ou 192.168.0.1).
    3. Connectez-vous avec vos identifiants administrateur.
    4. Recherchez une section appelée "Logs", "Journal", "Syslog" ou "Rapports".
    5. Vous pourrez généralement voir et télécharger les logs réseau.
    
    **Accès via SSH :**
    
    1. Utilisez un client SSH (comme PuTTY) pour vous connecter au dispositif.
    2. Entrez l'adresse IP du dispositif et vos identifiants administrateur.
    3. Utilisez des commandes spécifiques au dispositif pour afficher les logs (par exemple, `show log` sur certains routeurs).
    
    **Serveurs :**
    
    **Sous Windows :**
    
    1. Ouvrez l'Observateur d'événements (`eventvwr.msc`).
    2. Allez dans Journaux des applications et des services > Microsoft > Windows > Diagnostics-Performance > Opérationnel.
    3. Recherchez les événements liés au réseau.
    
    **Sous Linux et macOS :**
    
    1. Ouvrez un terminal.
    2. Accédez aux logs dans `/var/log` (par exemple, `cd /var/log`).
    3. Utilisez `cat`, `less` ou `grep` pour visualiser les logs, par exemple :
    ```bash
    cat /var/log/syslog | grep "network"
    ```
    
    **Systèmes de gestion de Logs centralisés :**
    
    - Exemples de systèmes : Splunk, ELK Stack (Elasticsearch, Logstash, Kibana), Graylog.
    - Connectez-vous à l'interface web de votre système de gestion de logs.
    - Utilisez les fonctionnalités de recherche et de filtrage pour trouver les informations réseau.
    """)

    st.markdown("""
    ### Guide pour accéder à ses principales informations de sécurité
    
    **Duration (duration) : La durée d'une connexion**
    
    - **Description :** La durée totale en secondes pendant laquelle une connexion réseau a été établie.
    - **Où la trouver :** Cette information peut être trouvée dans les journaux de connexion ou les fichiers de log de votre pare-feu ou de votre routeur.
    - **Comment l'obtenir :** Recherchez dans les logs des entrées contenant des informations sur le début et la fin des connexions. Soustrayez ces valeurs pour obtenir la durée.
    - **Exemple :** Si les logs indiquent start_time=10:00 et end_time=10:05, la durée est de 300 secondes.
    
    **Protocol Type (protocol_type) : Le protocole utilisé (TCP, UDP, ICMP)**
    
    - **Description :** Le type de protocole de communication utilisé par la connexion réseau.
    - **Où la trouver :** Généralement disponible dans les journaux de connexion ou les rapports de trafic réseau.
    - **Comment l'obtenir :** Les logs mentionnent souvent le type de protocole utilisé pour chaque connexion.
    - **Exemple :** Recherchez des entrées telles que protocol=TCP, protocol=UDP, ou protocol=ICMP.
    
    **Service (service) : Le service de destination (HTTP, FTP, SMTP, etc.)**
    
    - **Description :** Le service réseau auquel la connexion était destinée.
    - **Où la trouver :** Souvent indiqué dans les journaux de pare-feu ou les logs des serveurs.
    - **Comment l'obtenir :** Les logs indiquent souvent le service de destination pour chaque connexion.
    - **Exemple :** Recherchez des entrées telles que service=HTTP, service=FTP, ou service=SMTP.
    
    **Number of Failed Logins (num_failed_logins) : Le nombre de tentatives de connexion échouées**
    
    - **Description :** Le nombre de tentatives de connexion infructueuses avant la connexion réussie.
    - **Où la trouver :** Disponible dans les journaux d'authentification ou les fichiers de log de sécurité.
    - **Comment l'obtenir :** Les journaux d'authentification ou de sécurité enregistrent les tentatives de connexion échouées.
    - **Exemple :** Recherchez des entrées telles que failed login attempts=3.
    
    **Count (count) : Le nombre de connexions au même hôte**
    
    - **Description :** Le nombre de connexions établies au même hôte pendant une période donnée.
    - **Où la trouver :** Visible dans les journaux de connexion ou les rapports d'analyse de trafic.
    - **Comment l'obtenir :** Comptez le nombre de connexions au même hôte dans une période donnée en utilisant les logs de connexion.
    - **Exemple :** Si les logs montrent 5 connexions distinctes au même hôte en une heure, le count est 5.
    
    **Bytes envoyés et reçus (total_bytes) : Le volume de données échangées**
    
    - **Description :** La somme des octets envoyés et reçus pendant la connexion.
    - **Où la trouver :** Dans les logs de connexion réseau ou les outils de surveillance de bande passante.
    - **Comment l'obtenir :** Additionnez les octets envoyés (src_bytes) et reçus (dst_bytes) pour chaque connexion.
    - **Exemple :** Si src_bytes=1000 et dst_bytes=2000, alors total_bytes=3000.
    
    **État de connexion (logged_in) : Indique si la connexion a réussi ou non**
    
    - **Description :** Indique si la connexion a été réussie (1) ou échouée (0).
    - **Où la trouver :** Dans les journaux d'authentification ou les rapports de connexion.
    - **Comment l'obtenir :** Les journaux d'authentification indiquent souvent si la connexion a réussi ou non.
    - **Exemple :** Recherchez des entrées comme login successful=1 ou login failed=0.
    
    **Dst Host Same Src Port Rate (dst_host_same_src_port_rate) : Le taux de connexions au même port source**
    
    - **Description :** Le pourcentage de connexions au même port source sur l'hôte de destination.
    - **Où la trouver :** Nécessite une analyse des logs de connexion pour calculer cette métrique.
    - **Comment l'obtenir :** Calculez le pourcentage de connexions utilisant le même port source sur l'hôte de destination à partir des logs de connexion.
    - **Exemple :** Si sur 10 connexions, 7 utilisent le même port source, le taux est de 70%.
    
    ### Variables Moins Accessibles (Pour Information)
    
    **Srv Count (srv_count)**
    
    - **Description :** Le nombre de connexions au même service.
    - **Où la trouver :** Dans les logs de service ou les rapports d'analyse de trafic.
    
    **Flag (flag)**
    
    - **Description :** Indique l'état de la connexion TCP (par exemple, S0, REJ).
    - **Où la trouver :** Dans les logs de connexion TCP/IP.
    
    **Serror Rate (serror_rate) : Le taux d'erreurs de serveur**
    
    - **Description :** Le pourcentage de connexions ayant des erreurs de serveur.
    - **Où la trouver :** Nécessite une analyse des logs de connexion pour calculer cette métrique.
    
    **Rerror Rate (rerror_rate) : Le taux d'erreurs de réponse**
    
    - **Description :** Le pourcentage de connexions ayant des erreurs de réponse.
    - **Où la trouver :** Nécessite une analyse des logs de connexion pour calculer cette métrique.
    """)

    st.markdown("""
    ### Exemple de Journal de connexion
    
    Voici un exemple de ce à quoi peuvent ressembler les entrées de log :
    
    ```
    timestamp="2023-06-15T10:00:00Z" protocol="TCP" service="HTTP" src_bytes=500 dst_bytes=1500 duration=60 logged_in=1
    timestamp="2023-06-15T10:05:00Z" protocol="UDP" service="DNS" src_bytes=200 dst_bytes=300 duration=5 logged_in=0

        ```
    
    """)
    
    
# Préparation du modèle, 


# Charger les données et calculer les médianes
@st.cache_data
def load_training_data():
    data_train = pd.read_csv('KDDTrain+.txt', encoding='ISO-8859-1')
    columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
                'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
                'attack', 'level'])
    data_train.columns = columns
    data_train['total_bytes'] = data_train['src_bytes'] + data_train['dst_bytes']
    return data_train

data_train = load_training_data()

# Calculer les médianes des colonnes numériques 
numeric_columns = data_train.select_dtypes(include=['number']).columns
medians = data_train[numeric_columns].median()

# Définir les valeurs par défaut 
default_values = {
    "num_failed_logins": medians['num_failed_logins'],
    "dst_host_srv_serror_rate": medians['dst_host_srv_serror_rate'],
    "count": medians['count'],
    "serror_rate": medians['serror_rate'],
    "dst_host_diff_srv_rate": medians['dst_host_diff_srv_rate'],
    "dst_host_serror_rate": medians['dst_host_serror_rate'],
    "diff_srv_rate": medians['diff_srv_rate'],
    "dst_host_same_src_port_rate": medians['dst_host_same_src_port_rate'],
    "dst_host_count": medians['dst_host_count'],
    "dst_host_srv_diff_host_rate": medians['dst_host_srv_diff_host_rate'],
    "duration": medians['duration'],
    "total_bytes": medians['total_bytes'],
    "logged_in": medians['logged_in'],
    "land": medians['land'],
    "hot": medians['hot'],
    "same_srv_rate": medians['same_srv_rate'],
    "dst_host_srv_count": medians['dst_host_srv_count'],
    "wrong_fragment": medians['wrong_fragment'],
    "su_attempted": medians['su_attempted'],
    "protocol_type": "tcp",  
    "flag": "SF",
    "service": "http"
}

# Encoder
@st.cache_data
def fit_encoder(data):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(data[['protocol_type', 'flag', 'service']])
    return encoder

encoder = fit_encoder(data_train)

# Préparation du scaler à partir des données d'entraînement
@st.cache_data
def fit_scaler(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit(data)
    return scaler

scaler = fit_scaler(data_train)

# Charger le modèle
model = joblib.load('modelXGboost1.pkl')

# Fonction de prétraitement
def preprocess_data(data, scaler, encoder):
    selected_features = ['num_failed_logins', 'dst_host_srv_serror_rate', 'count', 'serror_rate', 'dst_host_diff_srv_rate',
                         'dst_host_serror_rate', 'diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_count',
                         'dst_host_srv_diff_host_rate', 'duration', 'total_bytes', 'logged_in', 'land', 'hot',
                         'same_srv_rate', 'dst_host_srv_count', 'wrong_fragment', 'su_attempted']
    data_scaled = pd.DataFrame([data])[selected_features]

    data_scaled = scaler.transform(data_scaled)

    encoded_features = encoder.transform(pd.DataFrame([data])[['protocol_type', 'flag', 'service']])
    encoded_columns = encoder.get_feature_names_out(['protocol_type', 'flag', 'service'])

    data_scaled_df = pd.DataFrame(data_scaled, columns=selected_features)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns)

    X = pd.concat([data_scaled_df, encoded_df], axis=1)

    return X

# Fonction de prédiction
def make_prediction(data, scaler, encoder, model):
    X = preprocess_data(data, scaler, encoder)
    predictions = model.predict(X)
    return predictions

# Interface utilisateur
st.title('Détection d\'Intrusion - Modèle de simulation')

st.markdown("<h3 style='color: #1F4E79;'>Saisie des Données au format JSON</h3>", unsafe_allow_html=True)

# JSON editor avec valeurs par défaut 
example_json = json.dumps(example_values, indent=4)
st.markdown(f"**Exemple de format JSON attendu :**\n```json\n{example_json}\n```")

data_json = st.text_area("Entrez vos données au format JSON ici:", json.dumps(default_values, indent=4))

try:
    data_dict = json.loads(data_json)
    for key, value in default_values.items():
        if key not in data_dict:
            data_dict[key] = value
    
    st.write("Données saisies :", data_dict)
    
    if st.button('Faire une prédiction'):
        predictions = make_prediction(data_dict, scaler, encoder, model)
        threshold = 0.4
        prediction_label = "Intrusion probable" if predictions[0] >= threshold else "Intrusion peu probable"
        st.write(f"Prédiction : {prediction_label} (Score: {predictions[0]:.4f})")
        st.markdown("**Plus votre score se rapproche de 1, plus l'intrusion est probable.** Le seuil a été fixé à 0,4 afin d'éviter au maximum les faux négatifs. Plus votre score est proche de 0, plus le risque qu'une intrusion soit en cours est faible.")
except json.JSONDecodeError:
    st.error("Le format JSON est invalide. Veuillez corriger les erreurs et réessayer.")

st.markdown("<h2 style='text-align: center; font-weight: bold;'>Merci de votre passage sur cette application !</h2>", unsafe_allow_html=True)

st.markdown("<p style='color: #4682B4; font-style: italic;'>Application, dashboard et modèle réalisés par Marine Fruitier en 2024.</p>", unsafe_allow_html=True)

