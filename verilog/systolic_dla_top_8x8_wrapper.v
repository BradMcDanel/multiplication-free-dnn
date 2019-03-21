    `define ADD_AXI_INTERFACE(NAME, IDX)     \
    input  [31:0] ``NAME``_``IDX``_s_axi_awaddr , \
    input  [ID_WIDTH-1:0] ``NAME``_``IDX``_s_axi_awid   , \
    input  [ 7:0] ``NAME``_``IDX``_s_axi_awlen  , \
    input  [ 2:0] ``NAME``_``IDX``_s_axi_awsize , \
    input  [ 1:0] ``NAME``_``IDX``_s_axi_awburst, \
    input         ``NAME``_``IDX``_s_axi_awlock , \
    input  [ 3:0] ``NAME``_``IDX``_s_axi_awcache, \
    input  [ 2:0] ``NAME``_``IDX``_s_axi_awprot , \
    input         ``NAME``_``IDX``_s_axi_awvalid, \
    output        ``NAME``_``IDX``_s_axi_awready, \
    input  [31:0] ``NAME``_``IDX``_s_axi_wdata  , \
    input  [ 3:0] ``NAME``_``IDX``_s_axi_wstrb  , \
    input         ``NAME``_``IDX``_s_axi_wlast  , \
    input         ``NAME``_``IDX``_s_axi_wvalid , \
    output        ``NAME``_``IDX``_s_axi_wready , \
    output reg [ID_WIDTH-1:0] ``NAME``_``IDX``_s_axi_bid = {(ID_WIDTH){1'b0}}   , \
    output [ 1:0] ``NAME``_``IDX``_s_axi_bresp  , \
    output        ``NAME``_``IDX``_s_axi_bvalid , \
    input         ``NAME``_``IDX``_s_axi_bready , \
    input  [31:0] ``NAME``_``IDX``_s_axi_araddr , \
    input  [ID_WIDTH-1:0] ``NAME``_``IDX``_s_axi_arid   , \
    input  [ 7:0] ``NAME``_``IDX``_s_axi_arlen  , \
    input  [ 2:0] ``NAME``_``IDX``_s_axi_arsize , \
    input  [ 1:0] ``NAME``_``IDX``_s_axi_arburst, \
    input         ``NAME``_``IDX``_s_axi_arlock , \
    input  [ 3:0] ``NAME``_``IDX``_s_axi_arcache, \
    input  [ 2:0] ``NAME``_``IDX``_s_axi_arprot , \
    input         ``NAME``_``IDX``_s_axi_arvalid, \
    output        ``NAME``_``IDX``_s_axi_arready, \
    output reg [ID_WIDTH-1:0] ``NAME``_``IDX``_s_axi_rid = {(ID_WIDTH){1'b0}}, \
    output [31:0] ``NAME``_``IDX``_s_axi_rdata  , \
    output [ 1:0] ``NAME``_``IDX``_s_axi_rresp  , \
    output        ``NAME``_``IDX``_s_axi_rlast  , \
    output        ``NAME``_``IDX``_s_axi_rvalid , \
    input         ``NAME``_``IDX``_s_axi_rready ,


    `define ADD_SRAM_INTERFACE(NAME, IDX, WIDTH, ADDR_W) \
   (*MARK_DEBUG="TRUE"*) wire               ``NAME``_``IDX``_bram_rst_a     ; \
   (*MARK_DEBUG="TRUE"*) wire               ``NAME``_``IDX``_bram_clk_a     ; \
   (*MARK_DEBUG="TRUE"*) wire               ``NAME``_``IDX``_bram_en_a      ; \
   (*MARK_DEBUG="TRUE"*) wire  [3:0]        ``NAME``_``IDX``_bram_we_a      ; \
   (*MARK_DEBUG="TRUE"*) wire  [ADDR_W-1:0] ``NAME``_``IDX``_bram_addr_a    ; \
   (*MARK_DEBUG="TRUE"*) wire  [WIDTH-1:0]  ``NAME``_``IDX``_bram_wrdata_a  ; \
   (*MARK_DEBUG="TRUE"*) wire  [WIDTH-1:0]  ``NAME``_``IDX``_bram_rddata_a  ;


`define ADD_AXI_TO_APB_INTERFACE \
    input  [15:0] s_axi_awaddr , \
    input  [ID_WIDTH-1:0]  s_axi_awid   , \
    input         s_axi_awvalid, \
    output        s_axi_awready, \
    input  [31:0] s_axi_wdata  , \
    input         s_axi_wvalid , \
    output        s_axi_wready , \
    output reg [ID_WIDTH-1:0] s_axi_bid = {(ID_WIDTH){1'b0}}, \
    output [ 1:0] s_axi_bresp  , \
    output        s_axi_bvalid , \
    input         s_axi_bready , \
    input  [15:0] s_axi_araddr , \
    input  [ID_WIDTH-1:0]  s_axi_arid   , \
    input         s_axi_arvalid, \
    output        s_axi_arready, \
    output reg [ID_WIDTH-1:0] s_axi_rid = {(ID_WIDTH){1'b0}}  , \
    output [31:0] s_axi_rdata  , \
    output [ 1:0] s_axi_rresp  , \
    output        s_axi_rvalid , \
    input         s_axi_rready ,

module systolic_dla_top_8x8_wrapper #(
    parameter           ID_WIDTH            = 2                                                                 ,
    parameter           SUBARRAY_WIDTH      = 8                                                                 ,
    parameter           SUBARRAY_HEIGHT     = 8                                                                 ,
    parameter           NUM_DATAFLOW_PER_MX = 8                                                                 ,
    parameter           W_DATA_MUX          = 3/*clog2(NUM_DATAFLOW_PER_MX)*/                                   ,
    parameter           WGT_SRAM_DEPTH      = 256                                                               ,
    parameter           WGT_SHIFT_WIDTH     = NUM_DATAFLOW_PER_MX                                               ,
    parameter           WGT_SRAM_ADDR_W     = 8/*clog2(WGT_SRAM_DEPTH)*/                                        ,
    parameter           ACC_SRAM_DEPTH      = 256*256                                                           ,
    parameter           ACC_SRAM_ADDR_W     = 16/*clog2(ACC_SRAM_DEPTH)*/                                       ,
    parameter           XIN_SRAM_DEPTH      = 256*256                                                           ,
    parameter           XIN_SRAM_ADDR_W     = 16/*clog2(XIN_SRAM_DEPTH)*/                                       ,
    parameter           N_XIN_PER_MX        = 8                                                                 ,
    parameter           N_ACC_PER_MX        = 32                                                                ,
    parameter           N_WGT_PER_MX        = 8                                                                 ,
    parameter           N_WGT_MX            = SUBARRAY_WIDTH / N_WGT_PER_MX                                     ,
    parameter           N_XIN_MX            = SUBARRAY_WIDTH * NUM_DATAFLOW_PER_MX/N_XIN_PER_MX                 ,
    parameter           N_ACC_MX            = SUBARRAY_HEIGHT * 4 / N_ACC_PER_MX
) (
    `ADD_AXI_TO_APB_INTERFACE
    // AXI2BRAM
    `ADD_AXI_INTERFACE(xin,   0) 
    `ADD_AXI_INTERFACE(xin,   1) 
    `ADD_AXI_INTERFACE(xin,   2) 
    `ADD_AXI_INTERFACE(xin,   3) 
    `ADD_AXI_INTERFACE(xin,   4) 
    `ADD_AXI_INTERFACE(xin,   5) 
    `ADD_AXI_INTERFACE(xin,   6) 
    `ADD_AXI_INTERFACE(xin,   7) 
    `ADD_AXI_INTERFACE(wgt,   0) 
    `ADD_AXI_INTERFACE(wgt,   1) 
    `ADD_AXI_INTERFACE(wgt,   2) 
    `ADD_AXI_INTERFACE(wgt,   3) 
    `ADD_AXI_INTERFACE(wgt,   4) 
    `ADD_AXI_INTERFACE(wgt,   5) 
    `ADD_AXI_INTERFACE(wgt,   6) 
    `ADD_AXI_INTERFACE(wgt,   7) 
    `ADD_AXI_INTERFACE(acc,   0) 
    `ADD_AXI_INTERFACE(deacc, 0) 
    input         s_axi_aclk   ,
    input         s_axi_aresetn
);

`ADD_SRAM_INTERFACE(xin,   0,  8, 16)
`ADD_SRAM_INTERFACE(xin,   1,  8, 16)
`ADD_SRAM_INTERFACE(xin,   2,  8, 16)
`ADD_SRAM_INTERFACE(xin,   3,  8, 16)
`ADD_SRAM_INTERFACE(xin,   4,  8, 16)
`ADD_SRAM_INTERFACE(xin,   5,  8, 16)
`ADD_SRAM_INTERFACE(xin,   6,  8, 16)
`ADD_SRAM_INTERFACE(xin,   7,  8, 16)
`ADD_SRAM_INTERFACE(wgt,   0,  8, 8 )
`ADD_SRAM_INTERFACE(wgt,   1,  8, 8 )
`ADD_SRAM_INTERFACE(wgt,   2,  8, 8 )
`ADD_SRAM_INTERFACE(wgt,   3,  8, 8 )
`ADD_SRAM_INTERFACE(wgt,   4,  8, 8 )
`ADD_SRAM_INTERFACE(wgt,   5,  8, 8 )
`ADD_SRAM_INTERFACE(wgt,   6,  8, 8 )
`ADD_SRAM_INTERFACE(wgt,   7,  8, 8 )
`ADD_SRAM_INTERFACE(acc,   0, 32, 16)
`ADD_SRAM_INTERFACE(deacc, 0, 32, 16)

    wire [15:0] m_apb_paddr  ;
    wire [ 0:0] m_apb_psel   ;
    wire        m_apb_penable;
    wire        m_apb_pwrite ;
    wire [31:0] m_apb_pwdata ;
    wire [ 0:0] m_apb_pready ;
    wire [31:0] m_apb_prdata ;
    wire [ 0:0] m_apb_pslverr;

axi_apb_bridge_0 u_axi_apb_bridge_0 (
    .s_axi_aclk   (s_axi_aclk   ),
    .s_axi_aresetn(s_axi_aresetn),
    .s_axi_awaddr (s_axi_awaddr ),
    .s_axi_awvalid(s_axi_awvalid),
    .s_axi_awready(s_axi_awready),
    .s_axi_wdata  (s_axi_wdata  ),
    .s_axi_wvalid (s_axi_wvalid ),
    .s_axi_wready (s_axi_wready ),
    .s_axi_bresp  (s_axi_bresp  ),
    .s_axi_bvalid (s_axi_bvalid ),
    .s_axi_bready (s_axi_bready ),
    .s_axi_araddr (s_axi_araddr ),
    .s_axi_arvalid(s_axi_arvalid),
    .s_axi_arready(s_axi_arready),
    .s_axi_rdata  (s_axi_rdata  ),
    .s_axi_rresp  (s_axi_rresp  ),
    .s_axi_rvalid (s_axi_rvalid ),
    .s_axi_rready (s_axi_rready ),
    .m_apb_paddr  (m_apb_paddr  ),
    .m_apb_psel   (m_apb_psel   ),
    .m_apb_penable(m_apb_penable),
    .m_apb_pwrite (m_apb_pwrite ),
    .m_apb_pwdata (m_apb_pwdata ),
    .m_apb_pready (m_apb_pready ),
    .m_apb_prdata (m_apb_prdata ),
    .m_apb_pslverr(m_apb_pslverr)
);

    (*MARK_DEBUG="TRUE"*) wire [                8*N_WGT_MX-1:0] wgt_sram_r_en   ;
                          wire [8*WGT_SRAM_ADDR_W*N_WGT_MX-1:0] wgt_sram_raddr  ;
                          wire [              8*8*N_WGT_MX-1:0] wgt_sram_rdata  ;
    (*MARK_DEBUG="TRUE"*) wire [                  N_XIN_MX-1:0] xin_sram_r_en   ;
                          wire [  XIN_SRAM_ADDR_W*N_XIN_MX-1:0] xin_sram_raddr  ;
                          wire [                8*N_XIN_MX-1:0] xin_sram_rdata  ;
    (*MARK_DEBUG="TRUE"*) wire [                  N_ACC_MX-1:0] acc_sram_r_en   ;
                          wire [  ACC_SRAM_ADDR_W*N_ACC_MX-1:0] acc_sram_raddr  ;
                          wire [               32*N_ACC_MX-1:0] acc_sram_rdata  ;
    (*MARK_DEBUG="TRUE"*) wire [                  N_ACC_MX-1:0] deacc_sram_w_en ;
                          wire [  ACC_SRAM_ADDR_W*N_ACC_MX-1:0] deacc_sram_waddr;
                          wire [               32*N_ACC_MX-1:0] deacc_sram_wdata;

    systolic_dla_top #(
        .SUBARRAY_WIDTH     (SUBARRAY_WIDTH     ),
        .SUBARRAY_HEIGHT    (SUBARRAY_HEIGHT    ),
        .NUM_DATAFLOW_PER_MX(NUM_DATAFLOW_PER_MX),
        .W_DATA_MUX         (W_DATA_MUX         ),
        .WGT_SRAM_DEPTH     (WGT_SRAM_DEPTH     ),
        .WGT_SHIFT_WIDTH    (WGT_SHIFT_WIDTH    ),
        .WGT_SRAM_ADDR_W    (WGT_SRAM_ADDR_W    ),
        .ACC_SRAM_DEPTH     (ACC_SRAM_DEPTH     ),
        .ACC_SRAM_ADDR_W    (ACC_SRAM_ADDR_W    ),
        .XIN_SRAM_DEPTH     (XIN_SRAM_DEPTH     ),
        .XIN_SRAM_ADDR_W    (XIN_SRAM_ADDR_W    ),
        .N_XIN_PER_MX       (N_XIN_PER_MX       ),
        .N_ACC_PER_MX       (N_ACC_PER_MX       ),
        .N_WGT_PER_MX       (N_WGT_PER_MX       ),
        .N_WGT_MX           (N_WGT_MX           ),
        .N_XIN_MX           (N_XIN_MX           ),
        .N_ACC_MX           (N_ACC_MX           )
    ) i_systolic_dla_top (
        .clk             (s_axi_aclk      ),
        .reset_n         (s_axi_aresetn   ),
        .psel            (m_apb_psel      ),
        .paddr           (m_apb_paddr     ),
        .pwrite          (m_apb_pwrite    ),
        .pwdata          (m_apb_pwdata    ),
        .penable         (m_apb_penable   ),
        .prdata          (m_apb_prdata    ),
        .pready          (m_apb_pready    ),
        .wgt_sram_r_en   (wgt_sram_r_en   ),
        .wgt_sram_raddr  (wgt_sram_raddr  ),
        .wgt_sram_rdata  (wgt_sram_rdata  ),
        .acc_sram_r_en   (acc_sram_r_en   ),
        .acc_sram_raddr  (acc_sram_raddr  ),
        .acc_sram_rdata  (acc_sram_rdata  ),
        .deacc_sram_w_en (deacc_sram_w_en ),
        .deacc_sram_waddr(deacc_sram_waddr),
        .deacc_sram_wdata(deacc_sram_wdata),
        .xin_sram_r_en   (xin_sram_r_en   ),
        .xin_sram_raddr  (xin_sram_raddr  ),
        .xin_sram_rdata  (xin_sram_rdata  )
    );

`define ADD_AXI2BRAM_RD_INSTANCE(NAME, IDX, SRAM_NAME, R_WIDTH, R_ADDR_WIDTH) \
                      wire        tmp_``NAME``_``IDX``_bram_rst_a   ;     \
                      wire        tmp_``NAME``_``IDX``_bram_clk_a   ;     \
(*MARK_DEBUG="TRUE"*) wire        tmp_``NAME``_``IDX``_bram_en_a    ;     \
(*MARK_DEBUG="TRUE"*) wire [3:0]  tmp_``NAME``_``IDX``_bram_we_a    ;     \
                      wire [17:0] tmp_``NAME``_``IDX``_bram_addr_a  ;     \
                      wire [31:0] tmp_``NAME``_``IDX``_bram_wrdata_a;     \
                      wire [31:0] tmp_``NAME``_``IDX``_bram_rddata_a;     \
axi_bram_ctrl_0 u_axi_bram_ctrl_``NAME``_``IDX`` (  \
    .s_axi_aclk   (s_axi_aclk                    ), \
    .s_axi_aresetn(s_axi_aresetn                 ), \
    .s_axi_awaddr (``NAME``_``IDX``_s_axi_awaddr ), \
    .s_axi_awlen  (``NAME``_``IDX``_s_axi_awlen  ), \
    .s_axi_awsize (``NAME``_``IDX``_s_axi_awsize ), \
    .s_axi_awburst(``NAME``_``IDX``_s_axi_awburst), \
    .s_axi_awlock (``NAME``_``IDX``_s_axi_awlock ), \
    .s_axi_awcache(``NAME``_``IDX``_s_axi_awcache), \
    .s_axi_awprot (``NAME``_``IDX``_s_axi_awprot ), \
    .s_axi_awvalid(``NAME``_``IDX``_s_axi_awvalid), \
    .s_axi_awready(``NAME``_``IDX``_s_axi_awready), \
    .s_axi_wdata  (``NAME``_``IDX``_s_axi_wdata  ), \
    .s_axi_wstrb  (``NAME``_``IDX``_s_axi_wstrb  ), \
    .s_axi_wlast  (``NAME``_``IDX``_s_axi_wlast  ), \
    .s_axi_wvalid (``NAME``_``IDX``_s_axi_wvalid ), \
    .s_axi_wready (``NAME``_``IDX``_s_axi_wready ), \
    .s_axi_bresp  (``NAME``_``IDX``_s_axi_bresp  ), \
    .s_axi_bvalid (``NAME``_``IDX``_s_axi_bvalid ), \
    .s_axi_bready (``NAME``_``IDX``_s_axi_bready ), \
    .s_axi_araddr (``NAME``_``IDX``_s_axi_araddr ), \
    .s_axi_arlen  (``NAME``_``IDX``_s_axi_arlen  ), \
    .s_axi_arsize (``NAME``_``IDX``_s_axi_arsize ), \
    .s_axi_arburst(``NAME``_``IDX``_s_axi_arburst), \
    .s_axi_arlock (``NAME``_``IDX``_s_axi_arlock ), \
    .s_axi_arcache(``NAME``_``IDX``_s_axi_arcache), \
    .s_axi_arprot (``NAME``_``IDX``_s_axi_arprot ), \
    .s_axi_arvalid(``NAME``_``IDX``_s_axi_arvalid), \
    .s_axi_arready(``NAME``_``IDX``_s_axi_arready), \
    .s_axi_rdata  (``NAME``_``IDX``_s_axi_rdata  ), \
    .s_axi_rresp  (``NAME``_``IDX``_s_axi_rresp  ), \
    .s_axi_rlast  (``NAME``_``IDX``_s_axi_rlast  ), \
    .s_axi_rvalid (``NAME``_``IDX``_s_axi_rvalid ), \
    .s_axi_rready (``NAME``_``IDX``_s_axi_rready ), \
    .bram_rst_a   (tmp_``NAME``_``IDX``_bram_rst_a   ), \
    .bram_clk_a   (tmp_``NAME``_``IDX``_bram_clk_a   ), \
    .bram_en_a    (tmp_``NAME``_``IDX``_bram_en_a    ), \
    .bram_we_a    (tmp_``NAME``_``IDX``_bram_we_a    ), \
    .bram_addr_a  (tmp_``NAME``_``IDX``_bram_addr_a  ), \
    .bram_wrdata_a(tmp_``NAME``_``IDX``_bram_wrdata_a), \
    .bram_rddata_a(tmp_``NAME``_``IDX``_bram_rddata_a)  \
    ); \
    sram #( \
        .DATA_WIDTH   (R_WIDTH          ), \
        .ADDR_WIDTH   (R_ADDR_WIDTH     ), \
        .RAM_SIZE     ((1<<R_ADDR_WIDTH))) \
        i_``NAME``_``IDX``_sram ( \
        .clk   (s_axi_aclk                     ), \
        .we    (``NAME``_``IDX``_bram_we_a     ), \
        .en    (``NAME``_``IDX``_bram_en_a     ), \
        .addr  (``NAME``_``IDX``_bram_addr_a   ), \
        .data_i(``NAME``_``IDX``_bram_wrdata_a ), \
        .data_o(``NAME``_``IDX``_bram_rddata_a ) \
    ); \
    assign ``NAME``_``IDX``_bram_rst_a    = s_axi_aresetn; \
    assign ``NAME``_``IDX``_bram_clk_a    = s_axi_aclk; \
    assign ``NAME``_``IDX``_bram_en_a     = tmp_``NAME``_``IDX``_bram_en_a | ``SRAM_NAME``_r_en; \
    assign ``NAME``_``IDX``_bram_we_a     = tmp_``NAME``_``IDX``_bram_we_a; \
    assign ``NAME``_``IDX``_bram_addr_a   = tmp_``NAME``_``IDX``_bram_en_a ? {tmp_``NAME``_``IDX``_bram_addr_a[2 +: R_ADDR_WIDTH]} : \
                                            ``SRAM_NAME``_raddr[R_ADDR_WIDTH*IDX +: R_ADDR_WIDTH]; \
    assign ``NAME``_``IDX``_bram_wrdata_a = tmp_``NAME``_``IDX``_bram_wrdata_a[R_WIDTH-1:0]; \
    assign tmp_``NAME``_``IDX``_bram_rddata_a = ``NAME``_``IDX``_bram_rddata_a; \
    assign ``SRAM_NAME``_rdata[R_WIDTH*IDX +: R_WIDTH] = ``NAME``_``IDX``_bram_rddata_a[R_WIDTH-1:0];


 `define ADD_AXI2BRAM_WR_INSTANCE(NAME, IDX, SRAM_NAME, W_WIDTH, W_ADDR_WIDTH) \
                       wire        tmp_``NAME``_``IDX``_bram_rst_a   ;     \
                       wire        tmp_``NAME``_``IDX``_bram_clk_a   ;     \
 (*MARK_DEBUG="TRUE"*) wire        tmp_``NAME``_``IDX``_bram_en_a    ;     \
 (*MARK_DEBUG="TRUE"*) wire [3:0]  tmp_``NAME``_``IDX``_bram_we_a    ;     \
                       wire [17:0] tmp_``NAME``_``IDX``_bram_addr_a  ;     \
                       wire [31:0] tmp_``NAME``_``IDX``_bram_wrdata_a;     \
                       wire [31:0] tmp_``NAME``_``IDX``_bram_rddata_a;     \
 axi_bram_ctrl_0 u_axi_bram_ctrl_``NAME``_``IDX`` (  \
     .s_axi_aclk   (s_axi_aclk                    ), \
     .s_axi_aresetn(s_axi_aresetn                 ), \
     .s_axi_awaddr ( ``NAME``_``IDX``_s_axi_awaddr  ), \
     .s_axi_awlen  ( ``NAME``_``IDX``_s_axi_awlen   ), \
     .s_axi_awsize ( ``NAME``_``IDX``_s_axi_awsize  ), \
     .s_axi_awburst( ``NAME``_``IDX``_s_axi_awburst ), \
     .s_axi_awlock ( ``NAME``_``IDX``_s_axi_awlock  ), \
     .s_axi_awcache( ``NAME``_``IDX``_s_axi_awcache ), \
     .s_axi_awprot ( ``NAME``_``IDX``_s_axi_awprot  ), \
     .s_axi_awvalid( ``NAME``_``IDX``_s_axi_awvalid ), \
     .s_axi_awready( ``NAME``_``IDX``_s_axi_awready ), \
     .s_axi_wdata  ( ``NAME``_``IDX``_s_axi_wdata   ), \
     .s_axi_wstrb  ( ``NAME``_``IDX``_s_axi_wstrb   ), \
     .s_axi_wlast  ( ``NAME``_``IDX``_s_axi_wlast   ), \
     .s_axi_wvalid ( ``NAME``_``IDX``_s_axi_wvalid  ), \
     .s_axi_wready ( ``NAME``_``IDX``_s_axi_wready  ), \
     .s_axi_bresp  ( ``NAME``_``IDX``_s_axi_bresp   ), \
     .s_axi_bvalid ( ``NAME``_``IDX``_s_axi_bvalid  ), \
     .s_axi_bready ( ``NAME``_``IDX``_s_axi_bready  ), \
     .s_axi_araddr ( ``NAME``_``IDX``_s_axi_araddr  ), \
     .s_axi_arlen  ( ``NAME``_``IDX``_s_axi_arlen   ), \
     .s_axi_arsize ( ``NAME``_``IDX``_s_axi_arsize  ), \
     .s_axi_arburst( ``NAME``_``IDX``_s_axi_arburst ), \
     .s_axi_arlock ( ``NAME``_``IDX``_s_axi_arlock  ), \
     .s_axi_arcache( ``NAME``_``IDX``_s_axi_arcache ), \
     .s_axi_arprot ( ``NAME``_``IDX``_s_axi_arprot  ), \
     .s_axi_arvalid( ``NAME``_``IDX``_s_axi_arvalid ), \
     .s_axi_arready( ``NAME``_``IDX``_s_axi_arready ), \
     .s_axi_rdata  ( ``NAME``_``IDX``_s_axi_rdata   ), \
     .s_axi_rresp  ( ``NAME``_``IDX``_s_axi_rresp   ), \
     .s_axi_rlast  ( ``NAME``_``IDX``_s_axi_rlast   ), \
     .s_axi_rvalid ( ``NAME``_``IDX``_s_axi_rvalid  ), \
     .s_axi_rready ( ``NAME``_``IDX``_s_axi_rready  ), \
     .bram_rst_a   (tmp_``NAME``_``IDX``_bram_rst_a   ), \
     .bram_clk_a   (tmp_``NAME``_``IDX``_bram_clk_a   ), \
     .bram_en_a    (tmp_``NAME``_``IDX``_bram_en_a    ), \
     .bram_we_a    (tmp_``NAME``_``IDX``_bram_we_a    ), \
     .bram_addr_a  (tmp_``NAME``_``IDX``_bram_addr_a  ), \
     .bram_wrdata_a(tmp_``NAME``_``IDX``_bram_wrdata_a), \
     .bram_rddata_a(tmp_``NAME``_``IDX``_bram_rddata_a)  \
     ); \
    sram #( \
        .DATA_WIDTH   (W_WIDTH          ), \
        .ADDR_WIDTH   (W_ADDR_WIDTH     ), \
        .RAM_SIZE     ((1<<W_ADDR_WIDTH))) \
        i_``NAME``_``IDX``_sram ( \
        .clk   (s_axi_aclk                     ), \
        .we    (``NAME``_``IDX``_bram_we_a     ), \
        .en    (``NAME``_``IDX``_bram_en_a     ), \
        .addr  (``NAME``_``IDX``_bram_addr_a   ), \
        .data_i(``NAME``_``IDX``_bram_wrdata_a ), \
        .data_o(``NAME``_``IDX``_bram_rddata_a ) \
    ); \
     assign ``NAME``_``IDX``_bram_rst_a    = s_axi_aresetn; \
     assign ``NAME``_``IDX``_bram_clk_a    = s_axi_aclk; \
     assign ``NAME``_``IDX``_bram_en_a     = tmp_``NAME``_``IDX``_bram_en_a | ``SRAM_NAME``_w_en; \
     assign ``NAME``_``IDX``_bram_we_a     = tmp_``NAME``_``IDX``_bram_en_a ? tmp_``NAME``_``IDX``_bram_we_a : \
                                             ``SRAM_NAME``_w_en; \
     assign ``NAME``_``IDX``_bram_addr_a   = tmp_``NAME``_``IDX``_bram_en_a ? {tmp_``NAME``_``IDX``_bram_addr_a[2 +: W_ADDR_WIDTH]} : \
                                             ``SRAM_NAME``_waddr[W_ADDR_WIDTH*IDX +: W_ADDR_WIDTH]; \
     assign ``NAME``_``IDX``_bram_wrdata_a = tmp_``NAME``_``IDX``_bram_en_a ? tmp_``NAME``_``IDX``_bram_wrdata_a[W_WIDTH-1:0] : \
                                             ``SRAM_NAME``_wdata[W_WIDTH*IDX +: W_WIDTH]; \
     assign tmp_``NAME``_``IDX``_bram_rddata_a = ``NAME``_``IDX``_bram_rddata_a;

    `ADD_AXI2BRAM_RD_INSTANCE (wgt, 0, wgt_sram, 8, 8)
    `ADD_AXI2BRAM_RD_INSTANCE (wgt, 1, wgt_sram, 8, 8)
    `ADD_AXI2BRAM_RD_INSTANCE (wgt, 2, wgt_sram, 8, 8)
    `ADD_AXI2BRAM_RD_INSTANCE (wgt, 3, wgt_sram, 8, 8)
    `ADD_AXI2BRAM_RD_INSTANCE (wgt, 4, wgt_sram, 8, 8)
    `ADD_AXI2BRAM_RD_INSTANCE (wgt, 5, wgt_sram, 8, 8)
    `ADD_AXI2BRAM_RD_INSTANCE (wgt, 6, wgt_sram, 8, 8)
    `ADD_AXI2BRAM_RD_INSTANCE (wgt, 7, wgt_sram, 8, 8)

    `ADD_AXI2BRAM_RD_INSTANCE (xin, 0, xin_sram, 8, 16)
    `ADD_AXI2BRAM_RD_INSTANCE (xin, 1, xin_sram, 8, 16)
    `ADD_AXI2BRAM_RD_INSTANCE (xin, 2, xin_sram, 8, 16)
    `ADD_AXI2BRAM_RD_INSTANCE (xin, 3, xin_sram, 8, 16)
    `ADD_AXI2BRAM_RD_INSTANCE (xin, 4, xin_sram, 8, 16)
    `ADD_AXI2BRAM_RD_INSTANCE (xin, 5, xin_sram, 8, 16)
    `ADD_AXI2BRAM_RD_INSTANCE (xin, 6, xin_sram, 8, 16)
    `ADD_AXI2BRAM_RD_INSTANCE (xin, 7, xin_sram, 8, 16)

    `ADD_AXI2BRAM_RD_INSTANCE (acc,   0, acc_sram,  32, 16)
    `ADD_AXI2BRAM_WR_INSTANCE (deacc, 0, deacc_sram,32, 16)

endmodule


