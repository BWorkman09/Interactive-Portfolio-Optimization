class UserPreferences:
    def __init__(self, risk_tolerance, max_investment, assets):
        self.risk_tolerance = risk_tolerance
        self.max_investment = max_investment
        self.assets = assets

self.returns = np.nan_to_num(self.returns, nan=0.0, posinf=0.0, neginf=0.0)
