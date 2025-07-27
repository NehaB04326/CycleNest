                    • Schedule a visit with your gynecologist\n
                    • Maintain a balanced diet rich in fiber\n
                    • Engage in regular physical activity\n
                    • Monitor your symptoms regularly
                    """
                else:
                    result = "Normal Scan Detected"
                    result_class = "normal"
                    tips = """
                    • Maintain a healthy lifestyle\n
                    • Schedule annual check-ups\n
                    • Monitor for any changes in symptoms
                    """
                
                st.markdown(
                    f'<p class="result {result_class}">{result}</p>', 
                    unsafe_allow_html=True
                )
                st.metric("Prediction Confidence", f"{max(confidence, 1-confidence):.1%}")
                
                with st.expander("Recommended Actions", expanded=True):
                    st.markdown(f'<p class="tips">{tips}</p>', unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
else:
    st.info("Please upload an ultrasound image to begin analysis.")

