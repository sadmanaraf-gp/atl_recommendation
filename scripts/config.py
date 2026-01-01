# List of features for the model
COLS = [
    'msisdn', 'total_dstr', 'voicerev_total', 'datarev_total',
    'mixed_bundle_rev', 'voicerev_trig', 'datarev_trig', 'vol_mb', 'mo_mou',
    'recharge_cnt', 'recharge_amount', 'recharge_max', 'data_rg_days',
    'voice_rg_days', 'smartphone', 'subs_all_90d',
    'days_since_last_rcrg', 'total_dstr_1', 'vol_mb_1',
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
    'rchg_chnl_unknown', 'segment_9box_data engaged','segment_9box_data only','segment_9box_data optimizer','segment_9box_low engaged','segment_9box_mature user',
    'segment_9box_premium','segment_9box_silent base','segment_9box_ultra premium','segment_9box_voice only'
]

FEATURES = [
    'total_dstr', 'voicerev_total', 'datarev_total',
    'mixed_bundle_rev', 'voicerev_trig', 'datarev_trig', 'vol_mb', 'mo_mou',
    'recharge_cnt', 'recharge_amount', 'recharge_max', 'data_rg_days',
    'voice_rg_days','days_since_last_rcrg','total_dstr_1', 'vol_mb_1',
    'mo_mou_1', 'recharge_cnt_1', 'voicerev_total_1', 'datarev_total_1',
    'mixed_bundle_rev_1', 'voicerev_trig_1', 'datarev_trig_1',
    'total_dstr_2', 'vol_mb_2', 'mo_mou_2', 'recharge_cnt_2',
    'voicerev_total_2', 'datarev_total_2', 'mixed_bundle_rev_2',
    'voicerev_trig_2', 'datarev_trig_2','recharge_max_1', 'recharge_max_2',
    'dstr_change_01', 'mou_change_01', 'vol_change_01', 'dstr_change_12',
    'mou_change_12', 'vol_change_12','circle_chittagong','circle_dhaka',
    'circle_khulna','circle_mymensingh','circle_rajshahi','circle_sylhet','circle_unknown','rchg_chnl_01. freq_retail_only',
    'rchg_chnl_02. freq_retail_dominant','rchg_chnl_03. freq_mixed','rchg_chnl_04. freq_digital_dominant','rchg_chnl_05. freq_digital_only',
    'rchg_chnl_unknown','segment_9box_data engaged','segment_9box_data only','segment_9box_data optimizer','segment_9box_low engaged','segment_9box_mature user',
    'segment_9box_premium','segment_9box_silent base','segment_9box_ultra premium','segment_9box_voice only'
]

TARGET = ['label']

# Model parameters
NODES = 2048
DROPOUT_RATE = 0.3
N_CLASSES = 51
EPOCHS = 20
BATCH_SIZE = 1024
VALIDATION_SPLIT = 0.3

# Oracle table names
TABLE_BASE_01 = "TBL_PL_BASE_01"
TABLE_BASE_02 = "TBL_PL_BASE_02"
TABLE_BASE_03 = "TBL_PL_BASE_03"
TABLE_INFER_01 = "TBL_PL_BASE_INFER_01"
TABLE_INFER_02 = "TBL_PL_BASE_INFER_02"
TABLE_INFER_03 = "TBL_PL_BASE_INFER_03"
TABLE_PRED_PREFIX = "TBL_PL_PRED_"
