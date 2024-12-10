def main():
    # ... [previous code remains the same]

    # Continuous camera input
    camera_placeholder = st.empty()
    predictions_placeholder = st.empty()

    # Add session state for tracking detection
    if 'last_detection_time' not in st.session_state:
        st.session_state.last_detection_time = 0

    camera_input = st.camera_input("Detect Car Parts", label_visibility="visible")
    
    if camera_input:
        # Calculate time since last detection
        current_time = time.time()
        time_since_last_detection = current_time - st.session_state.last_detection_time

        # Only process if enough time has passed based on frame_delay
        if time_since_last_detection >= frame_delay:
            image = Image.open(camera_input)
            
            with st.spinner("🔍 Analyzing..."):
                processed_img = preprocess_image(image)
                class_name, confidence, all_predictions = predict(model, processed_img)
                
                # Filter by confidence threshold
                if confidence >= confidence_threshold:
                    # Update last detection time
                    st.session_state.last_detection_time = current_time

                    # Draw prediction on image
                    annotated_image = draw_prediction_on_image(image, class_name, confidence)
                    camera_placeholder.image(annotated_image, caption="Current Detection")
                    
                    # Prepare and display top predictions
                    predictions_with_names = list(zip(class_names, all_predictions))
                    valid_predictions = [(name, prob) for name, prob in predictions_with_names if prob >= confidence_threshold]
                    sorted_predictions = sorted(valid_predictions, key=lambda x: x[1], reverse=True)[:5]
                    
                    # Display predictions
                    predictions_html = "<ul class='prediction-list'>"
                    for name, prob in sorted_predictions:
                        predictions_html += f'<li>{name}<span class="confidence">{prob:.1%}</span></li>'
                    predictions_html += '</ul>'
                    predictions_placeholder.markdown(predictions_html, unsafe_allow_html=True)
