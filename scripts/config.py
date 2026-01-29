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
        'srvc1', 'srvc2', 'srvc3', 'srvc4', 'srvc5', 'srvc6', 'srvc7', 'srvc8', 'srvc8','circle_chittagong','circle_dhaka',
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
N_CLASSES = 41
EPOCHS = 20
BATCH_SIZE = 1024
VALIDATION_SPLIT = 0.3
EXPORT_BATCH_SIZE=1559752
TAKER_PROB_CUTOFF = 0.05
NONTAKER_PROB_CUTOFF = 0.03
PACK_NO = 5

# Oracle table names
TABLE_BASE_01 = "TBL_ATL_BASE_01_SAMPLE"
TABLE_BASE_02 = "TBL_ATL_BASE_02_SAMPLE"
TABLE_BASE_03 = "TBL_ATL_BASE_03_SAMPLE"
TABLE_INFER_01 = "TBL_ATL_BASE_INFER_01_SAMPLE"
TABLE_INFER_02 = "TBL_ATL_BASE_INFER_02_SAMPLE"
TABLE_INFER_03 = "TBL_ATL_BASE_INFER_03_SAMPLE"
TABLE_PRED_PREFIX = "TBL_ATL_PRED_"
class_names= [20,29, 39, 49, 98, 99, 118, 119, 148, 158, 179, 197, 198, 208, 217, 219, 228, 249, 258, 279, 308, 309, 319,
 419, 499, 509, 519, 598, 599, 639, 698, 699, 729, 798, 818, 898, 989, 999, 1099, 1148, 1199]
