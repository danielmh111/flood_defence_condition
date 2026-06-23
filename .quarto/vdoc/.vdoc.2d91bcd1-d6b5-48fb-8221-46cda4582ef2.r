#
#
#
#
#
#
#
library("tidyr")
library("dplyr")
library("ggplot2")
library("arrow")
#
#
#
transitions_path <- file.path("data/processed/eir_inspection_transitions.parquet")
transition_df <- arrow::read_parquet(transitions_path)

dim(transition_df)

#
#
#
#

covs_path <- "data/processed/unified_aims_eir_bgs.parquet"
covs_df <- arrow::read_parquet(covs_path)

dim(covs_df)
#
#
#
#
head(transition_df)
head(covs_df)
#
#
#
#
transitions_full <- left_join(transition_df, covs_df, by = join_by(asset_id == aims__asset_id))

head(transitions_full)
#
#
#
#
#

hist <- ggplot(data=transitions_full, aes(x=interval_years)) +
    geom_histogram()

hist

#
#
#
#

tm <- transitions_full %>%
    ggplot(aes(grade_t0, grade_t1)) +
    geom_tile()

tm

#
#
#

tm <- transitions_full %>%
    ggplot(aes(grade_t0, grade_t1)) +
    geom_tile()

tm

#
#
#
