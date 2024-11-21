import math

def round_to_nice_digits(number):
    if number == 0:
        return 0  # Avoid log calculation for zero

    # Find the order of magnitude
    order_of_magnitude = math.floor(math.log10(abs(number)))

    # Calculate the number of significant decimal places to keep
    significant_places = max(1 - order_of_magnitude, 0)

    # Round the number to the calculated significant places
    rounded_number = round(number, significant_places)

    return rounded_number
