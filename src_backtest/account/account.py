from dataclasses import dataclass


@dataclass
class Account:
    capital: float

    def add_deduct_money(self, amount: float):
        """Adds or deduct money depending on if the amount is positive of negative"""
        self.capital += amount

    def bust(self):
        return self.capital <= 0
