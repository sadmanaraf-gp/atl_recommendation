# List of features for the model
COLS = ['msisdn', 'total_dstr', 'voicerev_total', 'datarev_total',
       'mixed_bundle_rev', 'voicerev_trig', 'datarev_trig', 'vol_mb', 'mo_mou',
       'recharge_cnt', 'recharge_amount', 'recharge_max', 'data_rg_days',
       'voice_rg_days', 'smartphone', 'subs_all_90d', 'mygp_m1_act','mygp_m2_act','mygp_active_days','mygp_pack_rev',
        'mygp_pack_hits','rchg_amt_max_mygp','rchg_amt_mygp','rchg_amt_max_digital','rchg_amt_digtal',
       'days_since_last_rcrg', 'total_dstr_1', 'vol_mb_1',
       'mo_mou_1', 'recharge_cnt_1', 'voicerev_total_1', 'datarev_total_1',
       'mixed_bundle_rev_1', 'voicerev_trig_1', 'datarev_trig_1',
       'total_dstr_2', 'vol_mb_2', 'mo_mou_2', 'recharge_cnt_2',
       'voicerev_total_2', 'datarev_total_2', 'mixed_bundle_rev_2',
       'voicerev_trig_2', 'datarev_trig_2','recharge_max_1', 'recharge_max_2', 'label', 
       'dstr_change_01', 'mou_change_01', 'vol_change_01', 'dstr_change_12',
       'mou_change_12', 'vol_change_12','amount','amount1','amount2','amount3','amount4','amount5','amount6','amount7','amount8',
       'srvc1', 'srvc2', 'srvc3', 'srvc4', 'srvc5', 'srvc6', 'srvc7', 'srvc8','circle_chittagong','circle_dhaka',
       'circle_khulna','circle_mymensingh','circle_rajshahi','circle_sylhet','circle_unknown','rchg_chnl_01. freq_retail_only',
       'rchg_chnl_02. freq_retail_dominant','rchg_chnl_03. freq_mixed','rchg_chnl_04. freq_digital_dominant','rchg_chnl_05. freq_digital_only',
       'rchg_chnl_unknown']

TAKER_FEATURES = ['total_dstr', 'voicerev_total', 'datarev_total',
       'mixed_bundle_rev', 'voicerev_trig', 'datarev_trig', 'vol_mb', 'mo_mou',
       'recharge_cnt', 'recharge_amount', 'recharge_max', 'data_rg_days',
       'voice_rg_days','days_since_last_rcrg','mygp_m1_act','mygp_m2_act','mygp_active_days',
        'mygp_pack_rev','mygp_pack_hits','rchg_amt_max_mygp','rchg_amt_mygp','rchg_amt_max_digital',
        'rchg_amt_digtal','total_dstr_1', 'vol_mb_1',       
       'mo_mou_1', 'recharge_cnt_1', 'voicerev_total_1', 'datarev_total_1',
       'mixed_bundle_rev_1', 'voicerev_trig_1', 'datarev_trig_1',
       'total_dstr_2', 'vol_mb_2', 'mo_mou_2', 'recharge_cnt_2',
       'voicerev_total_2', 'datarev_total_2', 'mixed_bundle_rev_2',
       'voicerev_trig_2', 'datarev_trig_2','recharge_max_1', 'recharge_max_2',
       'dstr_change_01', 'mou_change_01', 'vol_change_01', 'dstr_change_12',
       'mou_change_12', 'vol_change_12','amount1','amount2','amount3','amount4','amount5','amount6','amount7','amount8',
        'srvc1', 'srvc2', 'srvc3', 'srvc4', 'srvc5', 'srvc6', 'srvc7', 'srvc8','circle_chittagong','circle_dhaka',
       'circle_khulna','circle_mymensingh','circle_rajshahi','circle_sylhet','circle_unknown','rchg_chnl_01. freq_retail_only',
       'rchg_chnl_02. freq_retail_dominant','rchg_chnl_03. freq_mixed','rchg_chnl_04. freq_digital_dominant','rchg_chnl_05. freq_digital_only',
       'rchg_chnl_unknown']

NONTAKER_FEATURES = ['total_dstr', 'voicerev_total', 'datarev_total',
       'mixed_bundle_rev', 'voicerev_trig', 'datarev_trig', 'vol_mb', 'mo_mou',
       'recharge_cnt', 'recharge_amount', 'recharge_max', 'data_rg_days',
       'voice_rg_days','days_since_last_rcrg','mygp_m1_act','mygp_m2_act','mygp_active_days',
        'mygp_pack_rev','mygp_pack_hits','rchg_amt_max_mygp','rchg_amt_mygp','rchg_amt_max_digital',
        'rchg_amt_digtal','total_dstr_1', 'vol_mb_1',       
       'mo_mou_1', 'recharge_cnt_1', 'voicerev_total_1', 'datarev_total_1',
       'mixed_bundle_rev_1', 'voicerev_trig_1', 'datarev_trig_1',
       'total_dstr_2', 'vol_mb_2', 'mo_mou_2', 'recharge_cnt_2',
       'voicerev_total_2', 'datarev_total_2', 'mixed_bundle_rev_2',
       'voicerev_trig_2', 'datarev_trig_2','recharge_max_1', 'recharge_max_2',
       'dstr_change_01', 'mou_change_01', 'vol_change_01', 'dstr_change_12',
       'mou_change_12', 'vol_change_12','circle_chittagong','circle_dhaka',
       'circle_khulna','circle_mymensingh','circle_rajshahi','circle_sylhet','circle_unknown','rchg_chnl_01. freq_retail_only',
       'rchg_chnl_02. freq_retail_dominant','rchg_chnl_03. freq_mixed','rchg_chnl_04. freq_digital_dominant','rchg_chnl_05. freq_digital_only',
       'rchg_chnl_unknown']

TARGET = ['label']

# Model parameters
NODES = 2048
DROPOUT_RATE = 0.3
N_CLASSES = 51
EPOCHS = 20
BATCH_SIZE = 1024
VALIDATION_SPLIT = 0.3
EXPORT_BATCH_SIZE=1559752
TAKER_PROB_CUTOFF = 0.05
NONTAKER_PROB_CUTOFF = 0.03
PACK_NO = 5

# Loss function: set USE_FOCAL_LOSS=True for focal loss, False for label-smoothed crossentropy
USE_FOCAL_LOSS = True
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.25
LABEL_SMOOTHING = 0.1  # used only when USE_FOCAL_LOSS=False

# Non-taker proxy: bottom N% of subscribers by hit count
NONTAKER_HIT_QUANTILE = 0.3

# Oracle table names
TABLE_BASE_01 = "SA_TBL_ATL_BASE_01"
TABLE_BASE_02 = "SA_TBL_ATL_BASE_02"
TABLE_BASE_03 = "SA_TBL_ATL_BASE_03"
TABLE_INFER_01 = "SA_TBL_ATL_BASE_INFER_01"
TABLE_INFER_02 = "SA_TBL_ATL_BASE_INFER_02"
TABLE_INFER_03 = "SA_TBL_ATL_BASE_INFER_03"
TABLE_PRED_PREFIX = "SA_TBL_ATL_PRED_"
FINAL_TABLE = "SA_TBL_ATL_PRED_202603"
class_names= [19,29,39,49,59,68,89,98,99,107,108,118,119,129,139,
       148,158,179,197,198,208,217,219,228,249,258,268,279,308,309,319,
       398,399,419,498,499,519,598,599,639,649,698,699,729,798,818,898,899,999,1099,1199]
