# Model and Evaluation v2

The project no longer treats multiple bars after one crossover as independent observations. Every detected event has one row and one Event ID.

Walk-forward validation uses time-ordered expanding splits and a gap at least equal to the maximum forecast horizon. Metrics are reported for:

- all hourly rows;
- independent event rows;
- tradeability discrimination;
- probability calibration;
- cost-adjusted OOF paper trades;
- event types separately.

Qualification is per horizon. A weak 1h model no longer automatically disables a genuinely qualified 2h or 3h model. Live trading still requires the selected horizon itself to be qualified.

The metrics CSV columns are:

`horizon,samples,accuracy,balanced_accuracy,auc,event_samples,event_auc,tradeability_auc,calibration_error,selected_trades,mean_net_return,positive_fold_fraction`
