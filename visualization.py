import streamlit as st
import pandas as pd

from llama_index.experimental.param_tuner.base import RunResult, TunedResult

import pandas as pd
import streamlit as st
import altair as alt


def visualize_tuned_result(tuned_result):
    if not tuned_result.run_results:
        st.write("No results found. Please check the tuning process.")
        return

    best_result = tuned_result.best_run_result
    best_params = best_result.params
    best_score = best_result.score

    # Display best results
    st.write("Best results:")
    st.write(f"- Score: `{best_score}`")
    for param, value in best_params.items():
        if param in ("score", "chunk_size", "top_k"):
            st.write(f"- {param}: `{value}`")

    # Collecting data for visualization
    data = []
    param_keys = set()
    for result in tuned_result.run_results:
        row = {
            key: result.params.get(key)
            for key in result.params
            if key in ("score", "chunk_size", "top_k")
        }
        row["score"] = result.score
        data.append(row)
        param_keys.update(result.params.keys())

    if not data:
        st.write("No valid data available for visualization.")
        return

    df_results = pd.DataFrame(data)

    # Dynamic selection based on available parameter keys
    if not df_results.empty:
        selections = {}
        for key in ("chunk_size", "top_k"):
            options = sorted(df_results[key].unique())
            if len(options) > 1:
                selected_value = st.select_slider(f"Select {key}", options=options)
            else:
                selected_value = options[0]
            selections[key] = selected_value

        # Filter data based on selected values
        filtered_data = df_results
        for key, value in selections.items():
            filtered_data = filtered_data[filtered_data[key] == value]

        if not filtered_data.empty:
            st.write("Selected hyperparameter scores:", filtered_data)

            # Creating an Altair chart for the first selected parameter
            first_param = list(selections.keys())[0]
            chart = (
                alt.Chart(filtered_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X(f"{first_param}:N", title=f"{first_param.capitalize()}"),
                    y=alt.Y("score:Q", title="Score"),
                    tooltip=list(param_keys) + ["score"],
                    color=alt.Color(
                        f"{first_param}:N",
                        legend=alt.Legend(title=f"{first_param.capitalize()}"),
                    ),
                )
                .properties(
                    title=f"Performance by {first_param.capitalize()}",
                    width=600,
                    height=400,
                )
            )
            st.altair_chart(chart, use_container_width=True)

        else:
            st.write("No data available for selected hyperparameters.")

    st.write("All results:")
    st.write(df_results)
