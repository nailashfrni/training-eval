import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import plotly.io as pio
from evals.mmlu_eval import subject2category
pio.templates.default = "plotly_dark"

MODES = ['full', 'half', 'option only', 'question only']
GROUPS = ['stem', 'social science', 'humanities', 'other']
SUBJECTS = list(subject2category.keys())
MODE_COLORS = {
    'full': '#636EFA',      
    'half': '#EF553B',      
    'option only': '#00CC96',
    'question only': '#AB63FA'
}

st.title('MMLU Evaluation in GPT-4o-mini')

# Load result dataframes
dfs = {'original': {},
       'shuffled': {}}
PATH = 'result'
for filename in os.listdir(PATH):
    if filename.endswith('.csv'):
        filepath = os.path.join(PATH, filename)
        df = pd.read_csv(filepath)
        file_info = filename[:-4].split('_')
        if file_info[-1] == 'SHUFFLE':
            if len(file_info) == 6:
                mode = file_info[0]
            else:
                mode = file_info[0] + ' ' + file_info[1]
            dfs['shuffled'][mode] = df
        else:
            if len(file_info) == 5:
                mode = file_info[0]
            else:
                mode = file_info[0] + ' ' + file_info[1]
            dfs['original'][mode] = df

# Accuracy across mode (original, not shuffled)
accuracy_modes = {}
for m in MODES:
    accuracy_modes[m] = np.mean(dfs['original'][m]['Score'])

accuracy_df = pd.DataFrame({'mode': MODES, 'accuracy': list(accuracy_modes.values())})
fig = px.bar(
    accuracy_df,
    x="mode",
    y="accuracy", 
    color="mode",
    title="Accuracy Comparison Across Mode",
    labels={"mode": "Mode", "accuracy": "Accuracy"}
)
st.plotly_chart(fig, use_container_width=True)

# Accuracy per group across mode
group_scores_list = []
for m in MODES:
    group_scores = dfs['original'][m].groupby('Group').agg({"Score": "mean"}).reset_index()
    group_scores['Mode'] = m
    group_scores_list.append(group_scores)

group_scores_df = pd.concat(group_scores_list, ignore_index=True)    
fig = px.bar(
    group_scores_df,
    x="Group",
    y="Score",
    color="Mode",  
    barmode="group",  # Grouped bar chart
    title="Accuracy per Group Across Modes",
    labels={"Group": "Group", "Score": "Accuracy", "Mode": "Mode"},
)
st.plotly_chart(fig, use_container_width=True)
st.write('-> Tren di setiap grupnya sama')

st.subheader('Analysis per Mode')
mode_selection = st.pills("Mode", MODES, selection_mode="single")
if mode_selection:
    df = dfs['original'][mode_selection]
    overall_mean = np.mean(df['Score'].values)
    st.subheader(f'Overall Mean Accuracy: {overall_mean:.4f}')
    group_scores = df.groupby('Group').agg({"Score": "mean"})\
                    .reset_index().sort_values(by='Score', ascending=False)
    fig = px.bar(
        group_scores,
        x="Group",
        y="Score", 
        color=group_scores.index,
        title="Accuracy Comparison Across Groups",
        labels={"Group": "Group", "Score": "Accuracy"},
        color_discrete_map=MODE_COLORS
    )
    st.plotly_chart(fig, use_container_width=True)

    subject_scores = df.groupby('Subject').agg({"Score": "mean"})\
                    .reset_index().sort_values(by='Score', ascending=False)
    fig = px.bar(
        subject_scores.iloc[:10],
        x="Subject",
        y="Score", 
        color="Subject",
        title="Top 10 Subjects with Highest Accuracy",
        labels={"Subject": "Subject", "Score": "Accuracy"}
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        subject_scores.iloc[-10:],
        x="Subject",
        y="Score", 
        color="Subject",
        title="10 Subjects with Least Accuracy",
        labels={"Subject": "Subject", "Score": "Accuracy"}
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write('Average Accuracy for All Subjects')
    st.dataframe(subject_scores.reset_index(drop=True))

    st.subheader(f'Compare Subjects per Group ({mode_selection.capitalize()} Mode)')
    options = ["STEM", "Social Sciences", "Humanities", "Other"]
    selection = st.pills("Group", options, selection_mode="single")
    if selection:
        if selection == 'Social Sciences':
            selection = 'social_sciences'
        else:
            selection = selection.lower()
        group_accuracy = group_scores[group_scores['Group'] == selection]['Score'].iloc[0]
        st.write(f'Mean Accuracy: {group_accuracy:.4f}')
        filtered_df = df[df['Group'] == selection]
        filtered_subject_scores = filtered_df.groupby('Subject').agg({"Score": "mean"})\
                        .reset_index().sort_values(by='Score', ascending=False)
        fig = px.bar(
            filtered_subject_scores,
            x="Subject",
            y="Score", 
            color="Subject",
            labels={"Subject": "Subject", "Score": "Accuracy"}
        )
        st.plotly_chart(fig, use_container_width=True)


# Accuracy comparison overview before-after shuffle
accuracy_shuffle = []
for v in dfs.keys():
    for m in MODES:
        if m != 'question only':
            score = np.mean(dfs[v][m]['Score'])
            accuracy_shuffle.append([v, m, score])

accuracy_shuffle_df = pd.DataFrame(accuracy_shuffle, columns=['Version', 'Mode', 'Accuracy'])    
fig = px.bar(
    accuracy_shuffle_df,
    x="Mode",
    y="Accuracy",
    color="Version",  
    barmode="group",  # Grouped bar chart
    title="Accuracy Before-After Shuffle",
    labels={"Version": "Version", "Accuracy": "Accuracy", "Mode": "Mode"},
)
st.plotly_chart(fig, use_container_width=True)
st.write('Question only tidak ada karena tidak menggunakan option sehingga proses shuffle option tidak akan berpengaruh.')

st.subheader('Analysis per Subject')
mode_accuracy_df = pd.DataFrame()

for m in MODES:
    mode_df = dfs['original'][m]
    subject_scores = mode_df.groupby('Subject').agg({"Score": "mean"})\
                            .reset_index().sort_values(by='Score', ascending=False)
    subject_scores['Mode'] = m 
    mode_accuracy_df = pd.concat([mode_accuracy_df, subject_scores])
subject_selection = st.pills("Subject", SUBJECTS, selection_mode="single")
if subject_selection:
    subject_scores_filtered = mode_accuracy_df[mode_accuracy_df['Subject'] == subject_selection]
    # Create a bar chart comparing accuracy across modes for the selected subject
    fig = px.bar(
        subject_scores_filtered,
        x="Mode",
        y="Score", 
        color="Mode",
        title=f"Accuracy for {subject_selection.capitalize()} Across Modes",
        labels={"Mode": "Mode", "Score": "Accuracy"},
        color_discrete_map=MODE_COLORS
    )

    # Step 4: Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Optional: Display the average accuracy for the selected subject across all modes
    avg_accuracy = subject_scores_filtered['Score'].mean()
    st.write(f'Average accuracy for {subject_selection} across modes: {avg_accuracy:.4f}')