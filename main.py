from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import ctypes
import os
import pickle

app = Flask(__name__)

clf = pickle.load(open("hotel_model.pkl", 'rb'))
test_df = pickle.load(open("test_df.pkl", 'rb'))
int_cols = ['lead_time', 'arrival_date_week_number', 'stays_in_weekend_nights',
            'stays_in_week_nights', 'adults', 'children', 'is_repeated_guest',
            'previous_cancellations', 'booking_changes', 'days_in_waiting_list', 'adr']


@app.route('/', methods=["GET", "POST"])
@app.route('/home', methods=["GET", "POST"])
def home_page():
    if request.method == "POST":
        hotel = request.form.get("hotel")
        distribution_channel = request.form.get("distribution_channel")
        deposit_type = request.form.get("deposit_type")
        customer_type = request.form.get("customer_type")
        is_repeated_guest = request.form.get("is_repeated_guest")
        lead_time = request.form.get("lead_time")
        arrival_date_week_number = request.form.get("arrival_date_week_number")
        stays_in_weekend_nights = request.form.get("stays_in_weekend_nights")
        stays_in_week_nights = request.form.get("stays_in_week_nights")
        adults = request.form.get("adults")
        children = request.form.get("children")
        previous_cancellations = request.form.get("previous_cancellations")
        booking_changes = request.form.get("booking_changes")
        days_in_waiting_list = request.form.get("days_in_waiting_list")
        adr = request.form.get("adr")

        preds = predict_function(hotel, distribution_channel, deposit_type,
                                 customer_type, is_repeated_guest, lead_time,
                                 arrival_date_week_number, stays_in_weekend_nights,
                                 stays_in_week_nights, adults, children,
                                 previous_cancellations, booking_changes,
                                 days_in_waiting_list, adr)

        return render_template("predict.html", preds=preds)
        # Otherwise this was a normal GET request
    else:
        return render_template("home.html")


@app.route('/graphs', methods=["GET", "POST"])
def graph_page():
    return render_template("graphs.html")


def predict_function(hotel, distribution_channel, deposit_type,
                     customer_type, is_repeated_guest, lead_time, arrival_date_week_number, stays_in_weekend_nights,
                     stays_in_week_nights, adults, children,
                     previous_cancellations, booking_changes,
                     days_in_waiting_list, adr):
    user_input = {'hotel': hotel,
                  'lead_time': lead_time,
                  'arrival_date_week_number': arrival_date_week_number,
                  'stays_in_weekend_nights': stays_in_weekend_nights,
                  'stays_in_week_nights': stays_in_week_nights,
                  'adults': adults,
                  'children': children,
                  'distribution_channel': distribution_channel,
                  'is_repeated_guest': is_repeated_guest,
                  'previous_cancellations': previous_cancellations,
                  'booking_changes': booking_changes,
                  'deposit_type': deposit_type,
                  'days_in_waiting_list': days_in_waiting_list,
                  'customer_type': customer_type,
                  'adr': adr}

    user_input = pd.DataFrame([user_input])
    dummy_user_input_cols = list(set(user_input.columns) - set(int_cols))
    dummies_user_input = pd.get_dummies(user_input, columns=dummy_user_input_cols)
    result = pd.concat([test_df, dummies_user_input])
    result.fillna(0, inplace=True)
    preds = clf.predict(result)
    return preds


# only run app if this is run directly
if __name__ == "__main__":
    # start in debugging mode since we are making it
    app.run(debug=True)
