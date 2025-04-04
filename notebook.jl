### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ d5c91210-10a2-4e6f-8745-7acbfec5922b
# Install necessary packages
begin
    using Pkg
    Pkg.add(["Pluto", "CSV", "XLSX", "DataFrames", "Plots", "MLJ", "MLJLinearModels", "Statistics"])
end

# Load required libraries

# ╔═╡ 92f6d78e-7b22-4bf9-8141-27eab868764e
Pkg.add(["MLJ", "DecisionTree", "Surrogates", "Flux"])

# ╔═╡ fe57e09b-6af9-447b-ae70-5a9e5f48996a
begin
    using Pluto
    using CSV, XLSX, DataFrames
    using Plots
    using MLJ, MLJLinearModels
    using Statistics
end

# ╔═╡ a4e5ff89-8882-4237-a694-772f73135adf
using DecisionTree

# ╔═╡ d9f6fc25-7e42-48ae-9abd-050aae92f007
using Surrogates  # Corrected package for Random Forest

# ╔═╡ 998cdf9c-0429-45a9-b87d-1b7523843ffd
# Load dataset
begin
    df = DataFrame(CSV.File("C:\\Users\\Salatiel Johannes\\Documents\\Artificial Intelligence\\Assignment 1\\global_food_wastage_dataset.csv"))
    first(df, 5)  # Show first 5 rows
end

# ╔═╡ ada63a81-e9a4-4a56-ab35-7b50fd2857ee
#Inspect the dataset
println(first(df, 5))  # Display first 5 rows

# ╔═╡ de2084d2-f25a-4c2a-a072-899c13b91060
println(describe(df))  # Summary statistics

# ╔═╡ 0d023ba3-4eeb-4128-8775-38240538d395
# Check for missing values in each column and display the first 5 missing rows for each column
for col in names(df)
    missing_rows = findall(ismissing, df[!, col])  # Get indices of missing values
    if length(missing_rows) > 0
        println("Column: $col")
        println("First 5 missing values (rows): ", missing_rows[1:min(5, end)])
    end
end


# ╔═╡ 3e9121d8-d5b4-4c48-bc42-8d53c6a0382c
# Visualize target distribution
histogram(df."Economic Loss (Million \$)", label="Economic Loss")

# ╔═╡ 85454e3f-4d33-4b4d-9abb-f8d52c8bbc7d
xlabel!("Economic Loss")

# ╔═╡ 8c8bf0cb-751b-420e-8e63-3ca85fe1279e
ylabel!("Count")

# ╔═╡ 4c83882e-b276-4ef3-9395-0cbe1125770f
title!("Distribution of Economic Loss")

# ╔═╡ 721bc407-ead9-4be8-bdac-67beae0f28bc
# Selecting numeric columns only
numeric_columns = names(df)[map(c -> eltype(df[!, c]) <: Number, names(df))]

# ╔═╡ 30a06199-e38e-420a-8068-af688f24eeab
numeric_df = df[!, numeric_columns]

# ╔═╡ 9aa43388-3fdf-4bdf-9ffc-4c78537211e4
# Compute the correlation matrix for the numeric columns
correlation_matrix = cor(Matrix(numeric_df))

# ╔═╡ 3c1fc472-bc6a-494a-9801-8f39472b580b
# Plot the heatmap of the correlation matrix
heatmap(correlation_matrix, xlabel="Features", ylabel="Features", title="Correlation Matrix")

# ╔═╡ 79af78c7-685c-4b58-abf6-853a3fe1a949
# Example: Define a Decision Tree model
dt_model = DecisionTreeClassifier()

# ╔═╡ 38f8d770-7d61-40fd-9876-704dacfb7b12
# Example: Define a Random Forest model (using Surrogates)
rf_model = RandomForestRegressor()  # You can use RandomForestClassifier() if it's classification

# ╔═╡ 1705a2f8-8091-450b-9944-dfdfac22c631
model = Chain(
    Dense(4, 5, relu),  # First layer: 4 inputs, 5 hidden units, ReLU activation
    Dense(5, 1)         # Second layer: 5 hidden units, 1 output unit
)

# ╔═╡ e2ae27e6-cd1b-45a0-b952-0063514bc59a
# Show the model structure
model

# ╔═╡ 5f90d9a3-1b59-4e9e-bc33-5873b3042e3b

#Economic Loss Due to Fruit Wastage (Regression Model 1)
# Filter rows where Food category is "Fruits & Vegetables"
fruit_wastage_data = df[df[!, "Food Category"] .== "Fruits & Vegetables", :]

# Features: All columns except "Economic Loss (Million $)" and "Food category"

# ╔═╡ 92383c7a-08dc-442d-b3a7-ca79e7342782
X_fruit_wastage = select(fruit_wastage_data, Not(:"Economic Loss (Million \$)", "Food Category"))

# Target: Economic Loss

# ╔═╡ e9c11e77-7441-4c7e-8c4e-f27b20d26f9f
y_fruit_wastage = fruit_wastage_data[!, "Economic Loss (Million \$)"]

# ╔═╡ 073367bd-c0ce-4655-8034-299a89183f29
# Filter rows where Food category is "Dairy Products"
dairy_waste_data = df[df[!, "Food Category"] .== "Dairy Products", :]

# Features: All columns except "Household Waste" and "Food category"

# ╔═╡ a8c94992-9ab9-4cac-9752-097e08252414

X_dairy_waste = select(dairy_waste_data, Not([:Household Waste(%)"), :Food_category]))


# Target: Household Waste (assuming the column is Household Waste(%)")

# ╔═╡ a09f38b1-96b9-4ff5-9fa2-6e4298488894
y_dairy_waste = dairy_waste_data[!, :Household_Waste]

# ╔═╡ 5a9ce49e-dd0a-4a14-be91-81625e1a9cf4
#=╠═╡
using Flux

# Example: Define a simple neural network model with Flux
  ╠═╡ =#

# ╔═╡ d702bce2-5743-494a-89d6-61bdd9699258
# ╠═╡ disabled = true
#=╠═╡
using Flux
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═d5c91210-10a2-4e6f-8745-7acbfec5922b
# ╠═fe57e09b-6af9-447b-ae70-5a9e5f48996a
# ╠═998cdf9c-0429-45a9-b87d-1b7523843ffd
# ╠═ada63a81-e9a4-4a56-ab35-7b50fd2857ee
# ╠═de2084d2-f25a-4c2a-a072-899c13b91060
# ╠═0d023ba3-4eeb-4128-8775-38240538d395
# ╠═3e9121d8-d5b4-4c48-bc42-8d53c6a0382c
# ╠═85454e3f-4d33-4b4d-9abb-f8d52c8bbc7d
# ╠═8c8bf0cb-751b-420e-8e63-3ca85fe1279e
# ╠═4c83882e-b276-4ef3-9395-0cbe1125770f
# ╠═721bc407-ead9-4be8-bdac-67beae0f28bc
# ╠═30a06199-e38e-420a-8068-af688f24eeab
# ╠═9aa43388-3fdf-4bdf-9ffc-4c78537211e4
# ╠═3c1fc472-bc6a-494a-9801-8f39472b580b
# ╠═92f6d78e-7b22-4bf9-8141-27eab868764e
# ╠═a4e5ff89-8882-4237-a694-772f73135adf
# ╠═d9f6fc25-7e42-48ae-9abd-050aae92f007
# ╠═d702bce2-5743-494a-89d6-61bdd9699258
# ╠═79af78c7-685c-4b58-abf6-853a3fe1a949
# ╠═38f8d770-7d61-40fd-9876-704dacfb7b12
# ╠═5a9ce49e-dd0a-4a14-be91-81625e1a9cf4
# ╠═1705a2f8-8091-450b-9944-dfdfac22c631
# ╠═e2ae27e6-cd1b-45a0-b952-0063514bc59a
# ╠═5f90d9a3-1b59-4e9e-bc33-5873b3042e3b
# ╠═92383c7a-08dc-442d-b3a7-ca79e7342782
# ╠═e9c11e77-7441-4c7e-8c4e-f27b20d26f9f
# ╠═073367bd-c0ce-4655-8034-299a89183f29
# ╠═a8c94992-9ab9-4cac-9752-097e08252414
# ╠═a09f38b1-96b9-4ff5-9fa2-6e4298488894
