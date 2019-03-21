`timescale 1 ps / 1 ps

(* CHECK_LICENSE_TYPE = "axi_apb_bridge_0,axi_apb_bridge,{}" *) (* downgradeipidentifiedwarnings = "yes" *) (* x_core_info = "axi_apb_bridge,Vivado 2018.2" *) 
(* NotValidForBitStream *)
module axi_apb_bridge_0
   (s_axi_aclk,
    s_axi_aresetn,
    s_axi_awaddr,
    s_axi_awvalid,
    s_axi_awready,
    s_axi_wdata,
    s_axi_wvalid,
    s_axi_wready,
    s_axi_bresp,
    s_axi_bvalid,
    s_axi_bready,
    s_axi_araddr,
    s_axi_arvalid,
    s_axi_arready,
    s_axi_rdata,
    s_axi_rresp,
    s_axi_rvalid,
    s_axi_rready,
    m_apb_paddr,
    m_apb_psel,
    m_apb_penable,
    m_apb_pwrite,
    m_apb_pwdata,
    m_apb_pready,
    m_apb_prdata,
    m_apb_pslverr);
  (* x_interface_info = "xilinx.com:signal:clock:1.0 ACLK CLK" *) (* x_interface_parameter = "XIL_INTERFACENAME ACLK, ASSOCIATED_BUSIF AXI4_LITE:APB_M:APB_M2:APB_M3:APB_M4:APB_M5:APB_M6:APB_M7:APB_M8:APB_M9:APB_M10:APB_M11:APB_M12:APB_M13:APB_M14:APB_M15:APB_M16, ASSOCIATED_RESET s_axi_aresetn, FREQ_HZ 100000000, PHASE 0.000" *) input s_axi_aclk;
  (* x_interface_info = "xilinx.com:signal:reset:1.0 ARESETN RST" *) (* x_interface_parameter = "XIL_INTERFACENAME ARESETN, POLARITY ACTIVE_LOW" *) input s_axi_aresetn;
  (* x_interface_info = "xilinx.com:interface:aximm:1.0 AXI4_LITE AWADDR" *) (* x_interface_parameter = "XIL_INTERFACENAME AXI4_LITE, DATA_WIDTH 32, PROTOCOL AXI4LITE, FREQ_HZ 100000000, ID_WIDTH 0, ADDR_WIDTH 16, AWUSER_WIDTH 0, ARUSER_WIDTH 0, WUSER_WIDTH 0, RUSER_WIDTH 0, BUSER_WIDTH 0, READ_WRITE_MODE READ_WRITE, HAS_BURST 0, HAS_LOCK 0, HAS_PROT 0, HAS_CACHE 0, HAS_QOS 0, HAS_REGION 0, HAS_WSTRB 0, HAS_BRESP 1, HAS_RRESP 1, SUPPORTS_NARROW_BURST 0, NUM_READ_OUTSTANDING 1, NUM_WRITE_OUTSTANDING 1, MAX_BURST_LENGTH 1, PHASE 0.000, NUM_READ_THREADS 1, NUM_WRITE_THREADS 1, RUSER_BITS_PER_BYTE 0, WUSER_BITS_PER_BYTE 0" *) input [15:0]s_axi_awaddr;
  (* x_interface_info = "xilinx.com:interface:aximm:1.0 AXI4_LITE AWVALID" *) input s_axi_awvalid;
  (* x_interface_info = "xilinx.com:interface:aximm:1.0 AXI4_LITE AWREADY" *) output s_axi_awready;
  (* x_interface_info = "xilinx.com:interface:aximm:1.0 AXI4_LITE WDATA" *) input [31:0]s_axi_wdata;
  (* x_interface_info = "xilinx.com:interface:aximm:1.0 AXI4_LITE WVALID" *) input s_axi_wvalid;
  (* x_interface_info = "xilinx.com:interface:aximm:1.0 AXI4_LITE WREADY" *) output s_axi_wready;
  (* x_interface_info = "xilinx.com:interface:aximm:1.0 AXI4_LITE BRESP" *) output [1:0]s_axi_bresp;
  (* x_interface_info = "xilinx.com:interface:aximm:1.0 AXI4_LITE BVALID" *) output s_axi_bvalid;
  (* x_interface_info = "xilinx.com:interface:aximm:1.0 AXI4_LITE BREADY" *) input s_axi_bready;
  (* x_interface_info = "xilinx.com:interface:aximm:1.0 AXI4_LITE ARADDR" *) input [15:0]s_axi_araddr;
  (* x_interface_info = "xilinx.com:interface:aximm:1.0 AXI4_LITE ARVALID" *) input s_axi_arvalid;
  (* x_interface_info = "xilinx.com:interface:aximm:1.0 AXI4_LITE ARREADY" *) output s_axi_arready;
  (* x_interface_info = "xilinx.com:interface:aximm:1.0 AXI4_LITE RDATA" *) output [31:0]s_axi_rdata;
  (* x_interface_info = "xilinx.com:interface:aximm:1.0 AXI4_LITE RRESP" *) output [1:0]s_axi_rresp;
  (* x_interface_info = "xilinx.com:interface:aximm:1.0 AXI4_LITE RVALID" *) output s_axi_rvalid;
  (* x_interface_info = "xilinx.com:interface:aximm:1.0 AXI4_LITE RREADY" *) input s_axi_rready;
  (* x_interface_info = "xilinx.com:interface:apb:1.0 APB_M PADDR" *) output [15:0]m_apb_paddr;
  (* x_interface_info = "xilinx.com:interface:apb:1.0 APB_M PSEL" *) output [0:0]m_apb_psel;
  (* x_interface_info = "xilinx.com:interface:apb:1.0 APB_M PENABLE" *) output m_apb_penable;
  (* x_interface_info = "xilinx.com:interface:apb:1.0 APB_M PWRITE" *) output m_apb_pwrite;
  (* x_interface_info = "xilinx.com:interface:apb:1.0 APB_M PWDATA" *) output [31:0]m_apb_pwdata;
  (* x_interface_info = "xilinx.com:interface:apb:1.0 APB_M PREADY" *) input [0:0]m_apb_pready;
  (* x_interface_info = "xilinx.com:interface:apb:1.0 APB_M PRDATA" *) input [31:0]m_apb_prdata;
  (* x_interface_info = "xilinx.com:interface:apb:1.0 APB_M PSLVERR" *) input [0:0]m_apb_pslverr;

  wire [15:0]m_apb_paddr;
  wire m_apb_penable;
  wire [31:0]m_apb_prdata;
  wire [0:0]m_apb_pready;
  wire [0:0]m_apb_psel;
  wire [0:0]m_apb_pslverr;
  wire [31:0]m_apb_pwdata;
  wire m_apb_pwrite;
  wire s_axi_aclk;
  wire [15:0]s_axi_araddr;
  wire s_axi_aresetn;
  wire s_axi_arready;
  wire s_axi_arvalid;
  wire [15:0]s_axi_awaddr;
  wire s_axi_awready;
  wire s_axi_awvalid;
  wire s_axi_bready;
  wire [1:0]s_axi_bresp;
  wire s_axi_bvalid;
  wire [31:0]s_axi_rdata;
  wire s_axi_rready;
  wire [1:0]s_axi_rresp;
  wire s_axi_rvalid;
  wire [31:0]s_axi_wdata;
  wire s_axi_wready;
  wire s_axi_wvalid;
  wire [2:0]NLW_U0_m_apb_pprot_UNCONNECTED;
  wire [3:0]NLW_U0_m_apb_pstrb_UNCONNECTED;

  (* C_APB_NUM_SLAVES = "1" *) 
  (* C_BASEADDR = "64'b0000000000000000000000000000000000000000000000000000000000000000" *) 
  (* C_DPHASE_TIMEOUT = "16" *) 
  (* C_FAMILY = "artix7" *) 
  (* C_HIGHADDR = "64'b0000000000000000000000000000000000001111111111111111111111111111" *) 
  (* C_INSTANCE = "axi_apb_bridge_inst" *) 
  (* C_M_APB_ADDR_WIDTH = "16" *) 
  (* C_M_APB_DATA_WIDTH = "32" *) 
  (* C_M_APB_PROTOCOL = "apb3" *) 
  (* C_S_AXI_ADDR_WIDTH = "16" *) 
  (* C_S_AXI_DATA_WIDTH = "32" *) 
  (* C_S_AXI_RNG10_BASEADDR = "64'b0000000000000000000000000000000010010000000000000000000000000000" *) 
  (* C_S_AXI_RNG10_HIGHADDR = "64'b0000000000000000000000000000000010011111111111111111111111111111" *) 
  (* C_S_AXI_RNG11_BASEADDR = "64'b0000000000000000000000000000000010100000000000000000000000000000" *) 
  (* C_S_AXI_RNG11_HIGHADDR = "64'b0000000000000000000000000000000010101111111111111111111111111111" *) 
  (* C_S_AXI_RNG12_BASEADDR = "64'b0000000000000000000000000000000010110000000000000000000000000000" *) 
  (* C_S_AXI_RNG12_HIGHADDR = "64'b0000000000000000000000000000000010111111111111111111111111111111" *) 
  (* C_S_AXI_RNG13_BASEADDR = "64'b0000000000000000000000000000000011000000000000000000000000000000" *) 
  (* C_S_AXI_RNG13_HIGHADDR = "64'b0000000000000000000000000000000011001111111111111111111111111111" *) 
  (* C_S_AXI_RNG14_BASEADDR = "64'b0000000000000000000000000000000011010000000000000000000000000000" *) 
  (* C_S_AXI_RNG14_HIGHADDR = "64'b0000000000000000000000000000000011011111111111111111111111111111" *) 
  (* C_S_AXI_RNG15_BASEADDR = "64'b0000000000000000000000000000000011100000000000000000000000000000" *) 
  (* C_S_AXI_RNG15_HIGHADDR = "64'b0000000000000000000000000000000011101111111111111111111111111111" *) 
  (* C_S_AXI_RNG16_BASEADDR = "64'b0000000000000000000000000000000011110000000000000000000000000000" *) 
  (* C_S_AXI_RNG16_HIGHADDR = "64'b0000000000000000000000000000000011111111111111111111111111111111" *) 
  (* C_S_AXI_RNG2_BASEADDR = "64'b0000000000000000000000000000000000010000000000000000000000000000" *) 
  (* C_S_AXI_RNG2_HIGHADDR = "64'b0000000000000000000000000000000000011111111111111111111111111111" *) 
  (* C_S_AXI_RNG3_BASEADDR = "64'b0000000000000000000000000000000000100000000000000000000000000000" *) 
  (* C_S_AXI_RNG3_HIGHADDR = "64'b0000000000000000000000000000000000101111111111111111111111111111" *) 
  (* C_S_AXI_RNG4_BASEADDR = "64'b0000000000000000000000000000000000110000000000000000000000000000" *) 
  (* C_S_AXI_RNG4_HIGHADDR = "64'b0000000000000000000000000000000000111111111111111111111111111111" *) 
  (* C_S_AXI_RNG5_BASEADDR = "64'b0000000000000000000000000000000001000000000000000000000000000000" *) 
  (* C_S_AXI_RNG5_HIGHADDR = "64'b0000000000000000000000000000000001001111111111111111111111111111" *) 
  (* C_S_AXI_RNG6_BASEADDR = "64'b0000000000000000000000000000000001010000000000000000000000000000" *) 
  (* C_S_AXI_RNG6_HIGHADDR = "64'b0000000000000000000000000000000001011111111111111111111111111111" *) 
  (* C_S_AXI_RNG7_BASEADDR = "64'b0000000000000000000000000000000001100000000000000000000000000000" *) 
  (* C_S_AXI_RNG7_HIGHADDR = "64'b0000000000000000000000000000000001101111111111111111111111111111" *) 
  (* C_S_AXI_RNG8_BASEADDR = "64'b0000000000000000000000000000000001110000000000000000000000000000" *) 
  (* C_S_AXI_RNG8_HIGHADDR = "64'b0000000000000000000000000000000001111111111111111111111111111111" *) 
  (* C_S_AXI_RNG9_BASEADDR = "64'b0000000000000000000000000000000010000000000000000000000000000000" *) 
  (* C_S_AXI_RNG9_HIGHADDR = "64'b0000000000000000000000000000000010001111111111111111111111111111" *) 
  (* downgradeipidentifiedwarnings = "yes" *) 
  axi_apb_bridge_0_axi_apb_bridge U0
       (.m_apb_paddr(m_apb_paddr),
        .m_apb_penable(m_apb_penable),
        .m_apb_pprot(NLW_U0_m_apb_pprot_UNCONNECTED[2:0]),
        .m_apb_prdata(m_apb_prdata),
        .m_apb_prdata10({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .m_apb_prdata11({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .m_apb_prdata12({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .m_apb_prdata13({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .m_apb_prdata14({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .m_apb_prdata15({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .m_apb_prdata16({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .m_apb_prdata2({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .m_apb_prdata3({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .m_apb_prdata4({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .m_apb_prdata5({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .m_apb_prdata6({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .m_apb_prdata7({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .m_apb_prdata8({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .m_apb_prdata9({1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0,1'b0}),
        .m_apb_pready(m_apb_pready),
        .m_apb_psel(m_apb_psel),
        .m_apb_pslverr(m_apb_pslverr),
        .m_apb_pstrb(NLW_U0_m_apb_pstrb_UNCONNECTED[3:0]),
        .m_apb_pwdata(m_apb_pwdata),
        .m_apb_pwrite(m_apb_pwrite),
        .s_axi_aclk(s_axi_aclk),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_aresetn(s_axi_aresetn),
        .s_axi_arprot({1'b0,1'b0,1'b0}),
        .s_axi_arready(s_axi_arready),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awprot({1'b0,1'b0,1'b0}),
        .s_axi_awready(s_axi_awready),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rready(s_axi_rready),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wready(s_axi_wready),
        .s_axi_wstrb({1'b0,1'b0,1'b0,1'b0}),
        .s_axi_wvalid(s_axi_wvalid));
endmodule

(* ORIG_REF_NAME = "apb_mif" *) 
module axi_apb_bridge_0_apb_mif
   (out,
    m_apb_penable,
    m_apb_pwrite,
    PSEL_i,
    waddr_ready_sm1__0,
    slv_err_resp,
    p_1_in__0,
    m_apb_paddr,
    m_apb_pwdata,
    p_0_in,
    s_axi_aclk,
    \FSM_sequential_axi_wr_rd_cs_reg[1] ,
    apb_wr_request,
    dphase_timeout,
    m_apb_pready,
    s_axi_awvalid,
    s_axi_wvalid,
    m_apb_pslverr,
    apb_rd_request,
    D,
    E,
    \FSM_sequential_axi_wr_rd_cs_reg[0] );
  output [0:0]out;
  output m_apb_penable;
  output m_apb_pwrite;
  output PSEL_i;
  output waddr_ready_sm1__0;
  output slv_err_resp;
  output p_1_in__0;
  output [15:0]m_apb_paddr;
  output [31:0]m_apb_pwdata;
  input p_0_in;
  input s_axi_aclk;
  input \FSM_sequential_axi_wr_rd_cs_reg[1] ;
  input apb_wr_request;
  input dphase_timeout;
  input [0:0]m_apb_pready;
  input s_axi_awvalid;
  input s_axi_wvalid;
  input [0:0]m_apb_pslverr;
  input apb_rd_request;
  input [15:0]D;
  input [0:0]E;
  input [31:0]\FSM_sequential_axi_wr_rd_cs_reg[0] ;

  wire [15:0]D;
  wire [0:0]E;
  wire \FSM_onehot_apb_wr_rd_cs[0]_i_1_n_0 ;
  wire \FSM_onehot_apb_wr_rd_cs[1]_i_1_n_0 ;
  wire \FSM_onehot_apb_wr_rd_cs[2]_i_1_n_0 ;
  wire \FSM_onehot_apb_wr_rd_cs[2]_i_2_n_0 ;
  (* RTL_KEEP = "yes" *) wire \FSM_onehot_apb_wr_rd_cs_reg_n_0_[0] ;
  (* RTL_KEEP = "yes" *) wire \FSM_onehot_apb_wr_rd_cs_reg_n_0_[1] ;
  wire [31:0]\FSM_sequential_axi_wr_rd_cs_reg[0] ;
  wire \FSM_sequential_axi_wr_rd_cs_reg[1] ;
  wire PSEL_i;
  wire apb_penable_sm;
  wire apb_rd_request;
  wire apb_wr_request;
  wire dphase_timeout;
  wire [15:0]m_apb_paddr;
  wire m_apb_penable;
  wire [0:0]m_apb_pready;
  wire [0:0]m_apb_pslverr;
  wire [31:0]m_apb_pwdata;
  wire m_apb_pwrite;
  (* RTL_KEEP = "yes" *) wire [0:0]out;
  wire p_0_in;
  wire p_1_in__0;
  wire s_axi_aclk;
  wire s_axi_awvalid;
  wire s_axi_wvalid;
  wire slv_err_resp;
  wire waddr_ready_sm1__0;

  LUT4 #(
    .INIT(16'hA808)) 
    BRESP_1_i_i_2
       (.I0(out),
        .I1(dphase_timeout),
        .I2(m_apb_pready),
        .I3(m_apb_pslverr),
        .O(slv_err_resp));
  LUT3 #(
    .INIT(8'hF8)) 
    BVALID_i_i_2
       (.I0(m_apb_penable),
        .I1(m_apb_pready),
        .I2(dphase_timeout),
        .O(p_1_in__0));
  LUT6 #(
    .INIT(64'hAAABABABAAA8A8A8)) 
    \FSM_onehot_apb_wr_rd_cs[0]_i_1 
       (.I0(out),
        .I1(\FSM_onehot_apb_wr_rd_cs_reg_n_0_[1] ),
        .I2(\FSM_onehot_apb_wr_rd_cs[2]_i_2_n_0 ),
        .I3(\FSM_onehot_apb_wr_rd_cs_reg_n_0_[0] ),
        .I4(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .I5(\FSM_onehot_apb_wr_rd_cs_reg_n_0_[0] ),
        .O(\FSM_onehot_apb_wr_rd_cs[0]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hF1F1F1F1F0F0F0E0)) 
    \FSM_onehot_apb_wr_rd_cs[1]_i_1 
       (.I0(\FSM_onehot_apb_wr_rd_cs_reg_n_0_[1] ),
        .I1(\FSM_onehot_apb_wr_rd_cs[2]_i_2_n_0 ),
        .I2(\FSM_onehot_apb_wr_rd_cs_reg_n_0_[0] ),
        .I3(apb_rd_request),
        .I4(apb_wr_request),
        .I5(\FSM_onehot_apb_wr_rd_cs_reg_n_0_[1] ),
        .O(\FSM_onehot_apb_wr_rd_cs[1]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hABABABBBAAAAAAAA)) 
    \FSM_onehot_apb_wr_rd_cs[2]_i_1 
       (.I0(\FSM_onehot_apb_wr_rd_cs_reg_n_0_[1] ),
        .I1(\FSM_onehot_apb_wr_rd_cs[2]_i_2_n_0 ),
        .I2(\FSM_onehot_apb_wr_rd_cs_reg_n_0_[0] ),
        .I3(apb_rd_request),
        .I4(apb_wr_request),
        .I5(out),
        .O(\FSM_onehot_apb_wr_rd_cs[2]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hA8)) 
    \FSM_onehot_apb_wr_rd_cs[2]_i_2 
       (.I0(out),
        .I1(dphase_timeout),
        .I2(m_apb_pready),
        .O(\FSM_onehot_apb_wr_rd_cs[2]_i_2_n_0 ));
  (* FSM_ENCODED_STATES = "apb_idle:001,apb_setup:010,apb_access:100," *) 
  (* KEEP = "yes" *) 
  FDSE #(
    .INIT(1'b1)) 
    \FSM_onehot_apb_wr_rd_cs_reg[0] 
       (.C(s_axi_aclk),
        .CE(1'b1),
        .D(\FSM_onehot_apb_wr_rd_cs[0]_i_1_n_0 ),
        .Q(\FSM_onehot_apb_wr_rd_cs_reg_n_0_[0] ),
        .S(p_0_in));
  (* FSM_ENCODED_STATES = "apb_idle:001,apb_setup:010,apb_access:100," *) 
  (* KEEP = "yes" *) 
  FDRE #(
    .INIT(1'b0)) 
    \FSM_onehot_apb_wr_rd_cs_reg[1] 
       (.C(s_axi_aclk),
        .CE(1'b1),
        .D(\FSM_onehot_apb_wr_rd_cs[1]_i_1_n_0 ),
        .Q(\FSM_onehot_apb_wr_rd_cs_reg_n_0_[1] ),
        .R(p_0_in));
  (* FSM_ENCODED_STATES = "apb_idle:001,apb_setup:010,apb_access:100," *) 
  (* KEEP = "yes" *) 
  FDRE #(
    .INIT(1'b0)) 
    \FSM_onehot_apb_wr_rd_cs_reg[2] 
       (.C(s_axi_aclk),
        .CE(1'b1),
        .D(\FSM_onehot_apb_wr_rd_cs[2]_i_1_n_0 ),
        .Q(out),
        .R(p_0_in));
  LUT6 #(
    .INIT(64'hFFFFAAAEAAAEAAAE)) 
    \GEN_1_SELECT_SLAVE.M_APB_PSEL_i[0]_i_1 
       (.I0(\FSM_onehot_apb_wr_rd_cs_reg_n_0_[1] ),
        .I1(out),
        .I2(dphase_timeout),
        .I3(m_apb_pready),
        .I4(\FSM_onehot_apb_wr_rd_cs_reg_n_0_[0] ),
        .I5(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .O(PSEL_i));
  FDRE \PADDR_i_reg[0] 
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(D[0]),
        .Q(m_apb_paddr[0]),
        .R(p_0_in));
  FDRE \PADDR_i_reg[10] 
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(D[10]),
        .Q(m_apb_paddr[10]),
        .R(p_0_in));
  FDRE \PADDR_i_reg[11] 
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(D[11]),
        .Q(m_apb_paddr[11]),
        .R(p_0_in));
  FDRE \PADDR_i_reg[12] 
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(D[12]),
        .Q(m_apb_paddr[12]),
        .R(p_0_in));
  FDRE \PADDR_i_reg[13] 
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(D[13]),
        .Q(m_apb_paddr[13]),
        .R(p_0_in));
  FDRE \PADDR_i_reg[14] 
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(D[14]),
        .Q(m_apb_paddr[14]),
        .R(p_0_in));
  FDRE \PADDR_i_reg[15] 
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(D[15]),
        .Q(m_apb_paddr[15]),
        .R(p_0_in));
  FDRE \PADDR_i_reg[1] 
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(D[1]),
        .Q(m_apb_paddr[1]),
        .R(p_0_in));
  FDRE \PADDR_i_reg[2] 
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(D[2]),
        .Q(m_apb_paddr[2]),
        .R(p_0_in));
  FDRE \PADDR_i_reg[3] 
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(D[3]),
        .Q(m_apb_paddr[3]),
        .R(p_0_in));
  FDRE \PADDR_i_reg[4] 
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(D[4]),
        .Q(m_apb_paddr[4]),
        .R(p_0_in));
  FDRE \PADDR_i_reg[5] 
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(D[5]),
        .Q(m_apb_paddr[5]),
        .R(p_0_in));
  FDRE \PADDR_i_reg[6] 
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(D[6]),
        .Q(m_apb_paddr[6]),
        .R(p_0_in));
  FDRE \PADDR_i_reg[7] 
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(D[7]),
        .Q(m_apb_paddr[7]),
        .R(p_0_in));
  FDRE \PADDR_i_reg[8] 
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(D[8]),
        .Q(m_apb_paddr[8]),
        .R(p_0_in));
  FDRE \PADDR_i_reg[9] 
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(D[9]),
        .Q(m_apb_paddr[9]),
        .R(p_0_in));
  LUT4 #(
    .INIT(16'hABAA)) 
    PENABLE_i_i_1
       (.I0(\FSM_onehot_apb_wr_rd_cs_reg_n_0_[1] ),
        .I1(m_apb_pready),
        .I2(dphase_timeout),
        .I3(out),
        .O(apb_penable_sm));
  FDRE PENABLE_i_reg
       (.C(s_axi_aclk),
        .CE(1'b1),
        .D(apb_penable_sm),
        .Q(m_apb_penable),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[0] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [0]),
        .Q(m_apb_pwdata[0]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[10] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [10]),
        .Q(m_apb_pwdata[10]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[11] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [11]),
        .Q(m_apb_pwdata[11]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[12] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [12]),
        .Q(m_apb_pwdata[12]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[13] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [13]),
        .Q(m_apb_pwdata[13]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[14] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [14]),
        .Q(m_apb_pwdata[14]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[15] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [15]),
        .Q(m_apb_pwdata[15]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[16] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [16]),
        .Q(m_apb_pwdata[16]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[17] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [17]),
        .Q(m_apb_pwdata[17]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[18] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [18]),
        .Q(m_apb_pwdata[18]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[19] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [19]),
        .Q(m_apb_pwdata[19]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[1] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [1]),
        .Q(m_apb_pwdata[1]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[20] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [20]),
        .Q(m_apb_pwdata[20]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[21] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [21]),
        .Q(m_apb_pwdata[21]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[22] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [22]),
        .Q(m_apb_pwdata[22]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[23] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [23]),
        .Q(m_apb_pwdata[23]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[24] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [24]),
        .Q(m_apb_pwdata[24]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[25] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [25]),
        .Q(m_apb_pwdata[25]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[26] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [26]),
        .Q(m_apb_pwdata[26]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[27] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [27]),
        .Q(m_apb_pwdata[27]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[28] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [28]),
        .Q(m_apb_pwdata[28]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[29] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [29]),
        .Q(m_apb_pwdata[29]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[2] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [2]),
        .Q(m_apb_pwdata[2]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[30] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [30]),
        .Q(m_apb_pwdata[30]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[31] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [31]),
        .Q(m_apb_pwdata[31]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[3] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [3]),
        .Q(m_apb_pwdata[3]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[4] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [4]),
        .Q(m_apb_pwdata[4]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[5] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [5]),
        .Q(m_apb_pwdata[5]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[6] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [6]),
        .Q(m_apb_pwdata[6]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[7] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [7]),
        .Q(m_apb_pwdata[7]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[8] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [8]),
        .Q(m_apb_pwdata[8]),
        .R(p_0_in));
  FDRE \PWDATA_i_reg[9] 
       (.C(s_axi_aclk),
        .CE(E),
        .D(\FSM_sequential_axi_wr_rd_cs_reg[0] [9]),
        .Q(m_apb_pwdata[9]),
        .R(p_0_in));
  FDRE PWRITE_i_reg
       (.C(s_axi_aclk),
        .CE(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .D(apb_wr_request),
        .Q(m_apb_pwrite),
        .R(p_0_in));
  LUT2 #(
    .INIT(4'h8)) 
    WREADY_i_i_3
       (.I0(s_axi_awvalid),
        .I1(s_axi_wvalid),
        .O(waddr_ready_sm1__0));
endmodule

(* C_APB_NUM_SLAVES = "1" *) (* C_BASEADDR = "64'b0000000000000000000000000000000000000000000000000000000000000000" *) (* C_DPHASE_TIMEOUT = "16" *) 
(* C_FAMILY = "artix7" *) (* C_HIGHADDR = "64'b0000000000000000000000000000000000001111111111111111111111111111" *) (* C_INSTANCE = "axi_apb_bridge_inst" *) 
(* C_M_APB_ADDR_WIDTH = "16" *) (* C_M_APB_DATA_WIDTH = "32" *) (* C_M_APB_PROTOCOL = "apb3" *) 
(* C_S_AXI_ADDR_WIDTH = "16" *) (* C_S_AXI_DATA_WIDTH = "32" *) (* C_S_AXI_RNG10_BASEADDR = "64'b0000000000000000000000000000000010010000000000000000000000000000" *) 
(* C_S_AXI_RNG10_HIGHADDR = "64'b0000000000000000000000000000000010011111111111111111111111111111" *) (* C_S_AXI_RNG11_BASEADDR = "64'b0000000000000000000000000000000010100000000000000000000000000000" *) (* C_S_AXI_RNG11_HIGHADDR = "64'b0000000000000000000000000000000010101111111111111111111111111111" *) 
(* C_S_AXI_RNG12_BASEADDR = "64'b0000000000000000000000000000000010110000000000000000000000000000" *) (* C_S_AXI_RNG12_HIGHADDR = "64'b0000000000000000000000000000000010111111111111111111111111111111" *) (* C_S_AXI_RNG13_BASEADDR = "64'b0000000000000000000000000000000011000000000000000000000000000000" *) 
(* C_S_AXI_RNG13_HIGHADDR = "64'b0000000000000000000000000000000011001111111111111111111111111111" *) (* C_S_AXI_RNG14_BASEADDR = "64'b0000000000000000000000000000000011010000000000000000000000000000" *) (* C_S_AXI_RNG14_HIGHADDR = "64'b0000000000000000000000000000000011011111111111111111111111111111" *) 
(* C_S_AXI_RNG15_BASEADDR = "64'b0000000000000000000000000000000011100000000000000000000000000000" *) (* C_S_AXI_RNG15_HIGHADDR = "64'b0000000000000000000000000000000011101111111111111111111111111111" *) (* C_S_AXI_RNG16_BASEADDR = "64'b0000000000000000000000000000000011110000000000000000000000000000" *) 
(* C_S_AXI_RNG16_HIGHADDR = "64'b0000000000000000000000000000000011111111111111111111111111111111" *) (* C_S_AXI_RNG2_BASEADDR = "64'b0000000000000000000000000000000000010000000000000000000000000000" *) (* C_S_AXI_RNG2_HIGHADDR = "64'b0000000000000000000000000000000000011111111111111111111111111111" *) 
(* C_S_AXI_RNG3_BASEADDR = "64'b0000000000000000000000000000000000100000000000000000000000000000" *) (* C_S_AXI_RNG3_HIGHADDR = "64'b0000000000000000000000000000000000101111111111111111111111111111" *) (* C_S_AXI_RNG4_BASEADDR = "64'b0000000000000000000000000000000000110000000000000000000000000000" *) 
(* C_S_AXI_RNG4_HIGHADDR = "64'b0000000000000000000000000000000000111111111111111111111111111111" *) (* C_S_AXI_RNG5_BASEADDR = "64'b0000000000000000000000000000000001000000000000000000000000000000" *) (* C_S_AXI_RNG5_HIGHADDR = "64'b0000000000000000000000000000000001001111111111111111111111111111" *) 
(* C_S_AXI_RNG6_BASEADDR = "64'b0000000000000000000000000000000001010000000000000000000000000000" *) (* C_S_AXI_RNG6_HIGHADDR = "64'b0000000000000000000000000000000001011111111111111111111111111111" *) (* C_S_AXI_RNG7_BASEADDR = "64'b0000000000000000000000000000000001100000000000000000000000000000" *) 
(* C_S_AXI_RNG7_HIGHADDR = "64'b0000000000000000000000000000000001101111111111111111111111111111" *) (* C_S_AXI_RNG8_BASEADDR = "64'b0000000000000000000000000000000001110000000000000000000000000000" *) (* C_S_AXI_RNG8_HIGHADDR = "64'b0000000000000000000000000000000001111111111111111111111111111111" *) 
(* C_S_AXI_RNG9_BASEADDR = "64'b0000000000000000000000000000000010000000000000000000000000000000" *) (* C_S_AXI_RNG9_HIGHADDR = "64'b0000000000000000000000000000000010001111111111111111111111111111" *) (* ORIG_REF_NAME = "axi_apb_bridge" *) 
(* downgradeipidentifiedwarnings = "yes" *) 
module axi_apb_bridge_0_axi_apb_bridge
   (s_axi_aclk,
    s_axi_aresetn,
    s_axi_awaddr,
    s_axi_awprot,
    s_axi_awvalid,
    s_axi_awready,
    s_axi_wdata,
    s_axi_wstrb,
    s_axi_wvalid,
    s_axi_wready,
    s_axi_bresp,
    s_axi_bvalid,
    s_axi_bready,
    s_axi_araddr,
    s_axi_arprot,
    s_axi_arvalid,
    s_axi_arready,
    s_axi_rdata,
    s_axi_rresp,
    s_axi_rvalid,
    s_axi_rready,
    m_apb_paddr,
    m_apb_psel,
    m_apb_penable,
    m_apb_pwrite,
    m_apb_pwdata,
    m_apb_pready,
    m_apb_prdata,
    m_apb_prdata2,
    m_apb_prdata3,
    m_apb_prdata4,
    m_apb_prdata5,
    m_apb_prdata6,
    m_apb_prdata7,
    m_apb_prdata8,
    m_apb_prdata9,
    m_apb_prdata10,
    m_apb_prdata11,
    m_apb_prdata12,
    m_apb_prdata13,
    m_apb_prdata14,
    m_apb_prdata15,
    m_apb_prdata16,
    m_apb_pslverr,
    m_apb_pprot,
    m_apb_pstrb);
  input s_axi_aclk;
  input s_axi_aresetn;
  input [15:0]s_axi_awaddr;
  input [2:0]s_axi_awprot;
  input s_axi_awvalid;
  output s_axi_awready;
  input [31:0]s_axi_wdata;
  input [3:0]s_axi_wstrb;
  input s_axi_wvalid;
  output s_axi_wready;
  output [1:0]s_axi_bresp;
  output s_axi_bvalid;
  input s_axi_bready;
  input [15:0]s_axi_araddr;
  input [2:0]s_axi_arprot;
  input s_axi_arvalid;
  output s_axi_arready;
  output [31:0]s_axi_rdata;
  output [1:0]s_axi_rresp;
  output s_axi_rvalid;
  input s_axi_rready;
  output [15:0]m_apb_paddr;
  output [0:0]m_apb_psel;
  output m_apb_penable;
  output m_apb_pwrite;
  output [31:0]m_apb_pwdata;
  input [0:0]m_apb_pready;
  input [31:0]m_apb_prdata;
  input [31:0]m_apb_prdata2;
  input [31:0]m_apb_prdata3;
  input [31:0]m_apb_prdata4;
  input [31:0]m_apb_prdata5;
  input [31:0]m_apb_prdata6;
  input [31:0]m_apb_prdata7;
  input [31:0]m_apb_prdata8;
  input [31:0]m_apb_prdata9;
  input [31:0]m_apb_prdata10;
  input [31:0]m_apb_prdata11;
  input [31:0]m_apb_prdata12;
  input [31:0]m_apb_prdata13;
  input [31:0]m_apb_prdata14;
  input [31:0]m_apb_prdata15;
  input [31:0]m_apb_prdata16;
  input [0:0]m_apb_pslverr;
  output [2:0]m_apb_pprot;
  output [3:0]m_apb_pstrb;

  wire \<const0> ;
  wire \<const1> ;
  wire APB_MASTER_IF_MODULE_n_0;
  wire AXILITE_SLAVE_IF_MODULE_n_11;
  wire AXILITE_SLAVE_IF_MODULE_n_12;
  wire AXILITE_SLAVE_IF_MODULE_n_13;
  wire AXILITE_SLAVE_IF_MODULE_n_14;
  wire AXILITE_SLAVE_IF_MODULE_n_15;
  wire AXILITE_SLAVE_IF_MODULE_n_16;
  wire AXILITE_SLAVE_IF_MODULE_n_17;
  wire AXILITE_SLAVE_IF_MODULE_n_18;
  wire AXILITE_SLAVE_IF_MODULE_n_19;
  wire AXILITE_SLAVE_IF_MODULE_n_20;
  wire AXILITE_SLAVE_IF_MODULE_n_21;
  wire AXILITE_SLAVE_IF_MODULE_n_22;
  wire AXILITE_SLAVE_IF_MODULE_n_23;
  wire AXILITE_SLAVE_IF_MODULE_n_24;
  wire AXILITE_SLAVE_IF_MODULE_n_25;
  wire AXILITE_SLAVE_IF_MODULE_n_26;
  wire AXILITE_SLAVE_IF_MODULE_n_27;
  wire AXILITE_SLAVE_IF_MODULE_n_28;
  wire AXILITE_SLAVE_IF_MODULE_n_29;
  wire AXILITE_SLAVE_IF_MODULE_n_30;
  wire AXILITE_SLAVE_IF_MODULE_n_31;
  wire AXILITE_SLAVE_IF_MODULE_n_32;
  wire AXILITE_SLAVE_IF_MODULE_n_33;
  wire AXILITE_SLAVE_IF_MODULE_n_34;
  wire AXILITE_SLAVE_IF_MODULE_n_35;
  wire AXILITE_SLAVE_IF_MODULE_n_36;
  wire AXILITE_SLAVE_IF_MODULE_n_37;
  wire AXILITE_SLAVE_IF_MODULE_n_38;
  wire AXILITE_SLAVE_IF_MODULE_n_39;
  wire AXILITE_SLAVE_IF_MODULE_n_40;
  wire AXILITE_SLAVE_IF_MODULE_n_41;
  wire AXILITE_SLAVE_IF_MODULE_n_42;
  wire AXILITE_SLAVE_IF_MODULE_n_43;
  wire AXILITE_SLAVE_IF_MODULE_n_44;
  wire AXILITE_SLAVE_IF_MODULE_n_45;
  wire AXILITE_SLAVE_IF_MODULE_n_46;
  wire AXILITE_SLAVE_IF_MODULE_n_47;
  wire AXILITE_SLAVE_IF_MODULE_n_48;
  wire AXILITE_SLAVE_IF_MODULE_n_49;
  wire AXILITE_SLAVE_IF_MODULE_n_50;
  wire AXILITE_SLAVE_IF_MODULE_n_51;
  wire AXILITE_SLAVE_IF_MODULE_n_52;
  wire AXILITE_SLAVE_IF_MODULE_n_53;
  wire AXILITE_SLAVE_IF_MODULE_n_54;
  wire AXILITE_SLAVE_IF_MODULE_n_55;
  wire AXILITE_SLAVE_IF_MODULE_n_56;
  wire AXILITE_SLAVE_IF_MODULE_n_57;
  wire AXILITE_SLAVE_IF_MODULE_n_58;
  wire AXILITE_SLAVE_IF_MODULE_n_59;
  wire AXILITE_SLAVE_IF_MODULE_n_60;
  wire PSEL_i;
  wire apb_rd_request;
  wire apb_wr_request;
  wire dphase_timeout;
  wire [15:0]m_apb_paddr;
  wire m_apb_penable;
  wire [31:0]m_apb_prdata;
  wire [0:0]m_apb_pready;
  wire [0:0]m_apb_psel;
  wire [0:0]m_apb_pslverr;
  wire [31:0]m_apb_pwdata;
  wire m_apb_pwrite;
  wire p_0_in;
  wire p_1_in__0;
  wire s_axi_aclk;
  wire [15:0]s_axi_araddr;
  wire s_axi_aresetn;
  wire s_axi_arready;
  wire s_axi_arvalid;
  wire [15:0]s_axi_awaddr;
  wire s_axi_awready;
  wire s_axi_awvalid;
  wire s_axi_bready;
  wire [1:1]\^s_axi_bresp ;
  wire s_axi_bvalid;
  wire [31:0]s_axi_rdata;
  wire s_axi_rready;
  wire [1:1]\^s_axi_rresp ;
  wire s_axi_rvalid;
  wire [31:0]s_axi_wdata;
  wire s_axi_wready;
  wire s_axi_wvalid;
  wire slv_err_resp;
  wire waddr_ready_sm1__0;

  assign m_apb_pprot[2] = \<const0> ;
  assign m_apb_pprot[1] = \<const0> ;
  assign m_apb_pprot[0] = \<const0> ;
  assign m_apb_pstrb[3] = \<const1> ;
  assign m_apb_pstrb[2] = \<const1> ;
  assign m_apb_pstrb[1] = \<const1> ;
  assign m_apb_pstrb[0] = \<const1> ;
  assign s_axi_bresp[1] = \^s_axi_bresp [1];
  assign s_axi_bresp[0] = \<const0> ;
  assign s_axi_rresp[1] = \^s_axi_rresp [1];
  assign s_axi_rresp[0] = \<const0> ;
  axi_apb_bridge_0_apb_mif APB_MASTER_IF_MODULE
       (.D({AXILITE_SLAVE_IF_MODULE_n_13,AXILITE_SLAVE_IF_MODULE_n_14,AXILITE_SLAVE_IF_MODULE_n_15,AXILITE_SLAVE_IF_MODULE_n_16,AXILITE_SLAVE_IF_MODULE_n_17,AXILITE_SLAVE_IF_MODULE_n_18,AXILITE_SLAVE_IF_MODULE_n_19,AXILITE_SLAVE_IF_MODULE_n_20,AXILITE_SLAVE_IF_MODULE_n_21,AXILITE_SLAVE_IF_MODULE_n_22,AXILITE_SLAVE_IF_MODULE_n_23,AXILITE_SLAVE_IF_MODULE_n_24,AXILITE_SLAVE_IF_MODULE_n_25,AXILITE_SLAVE_IF_MODULE_n_26,AXILITE_SLAVE_IF_MODULE_n_27,AXILITE_SLAVE_IF_MODULE_n_28}),
        .E(AXILITE_SLAVE_IF_MODULE_n_11),
        .\FSM_sequential_axi_wr_rd_cs_reg[0] ({AXILITE_SLAVE_IF_MODULE_n_29,AXILITE_SLAVE_IF_MODULE_n_30,AXILITE_SLAVE_IF_MODULE_n_31,AXILITE_SLAVE_IF_MODULE_n_32,AXILITE_SLAVE_IF_MODULE_n_33,AXILITE_SLAVE_IF_MODULE_n_34,AXILITE_SLAVE_IF_MODULE_n_35,AXILITE_SLAVE_IF_MODULE_n_36,AXILITE_SLAVE_IF_MODULE_n_37,AXILITE_SLAVE_IF_MODULE_n_38,AXILITE_SLAVE_IF_MODULE_n_39,AXILITE_SLAVE_IF_MODULE_n_40,AXILITE_SLAVE_IF_MODULE_n_41,AXILITE_SLAVE_IF_MODULE_n_42,AXILITE_SLAVE_IF_MODULE_n_43,AXILITE_SLAVE_IF_MODULE_n_44,AXILITE_SLAVE_IF_MODULE_n_45,AXILITE_SLAVE_IF_MODULE_n_46,AXILITE_SLAVE_IF_MODULE_n_47,AXILITE_SLAVE_IF_MODULE_n_48,AXILITE_SLAVE_IF_MODULE_n_49,AXILITE_SLAVE_IF_MODULE_n_50,AXILITE_SLAVE_IF_MODULE_n_51,AXILITE_SLAVE_IF_MODULE_n_52,AXILITE_SLAVE_IF_MODULE_n_53,AXILITE_SLAVE_IF_MODULE_n_54,AXILITE_SLAVE_IF_MODULE_n_55,AXILITE_SLAVE_IF_MODULE_n_56,AXILITE_SLAVE_IF_MODULE_n_57,AXILITE_SLAVE_IF_MODULE_n_58,AXILITE_SLAVE_IF_MODULE_n_59,AXILITE_SLAVE_IF_MODULE_n_60}),
        .\FSM_sequential_axi_wr_rd_cs_reg[1] (AXILITE_SLAVE_IF_MODULE_n_12),
        .PSEL_i(PSEL_i),
        .apb_rd_request(apb_rd_request),
        .apb_wr_request(apb_wr_request),
        .dphase_timeout(dphase_timeout),
        .m_apb_paddr(m_apb_paddr),
        .m_apb_penable(m_apb_penable),
        .m_apb_pready(m_apb_pready),
        .m_apb_pslverr(m_apb_pslverr),
        .m_apb_pwdata(m_apb_pwdata),
        .m_apb_pwrite(m_apb_pwrite),
        .out(APB_MASTER_IF_MODULE_n_0),
        .p_0_in(p_0_in),
        .p_1_in__0(p_1_in__0),
        .s_axi_aclk(s_axi_aclk),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_wvalid(s_axi_wvalid),
        .slv_err_resp(slv_err_resp),
        .waddr_ready_sm1__0(waddr_ready_sm1__0));
  axi_apb_bridge_0_axilite_sif AXILITE_SLAVE_IF_MODULE
       (.D({AXILITE_SLAVE_IF_MODULE_n_13,AXILITE_SLAVE_IF_MODULE_n_14,AXILITE_SLAVE_IF_MODULE_n_15,AXILITE_SLAVE_IF_MODULE_n_16,AXILITE_SLAVE_IF_MODULE_n_17,AXILITE_SLAVE_IF_MODULE_n_18,AXILITE_SLAVE_IF_MODULE_n_19,AXILITE_SLAVE_IF_MODULE_n_20,AXILITE_SLAVE_IF_MODULE_n_21,AXILITE_SLAVE_IF_MODULE_n_22,AXILITE_SLAVE_IF_MODULE_n_23,AXILITE_SLAVE_IF_MODULE_n_24,AXILITE_SLAVE_IF_MODULE_n_25,AXILITE_SLAVE_IF_MODULE_n_26,AXILITE_SLAVE_IF_MODULE_n_27,AXILITE_SLAVE_IF_MODULE_n_28}),
        .E(AXILITE_SLAVE_IF_MODULE_n_11),
        .\PADDR_i_reg[15] (AXILITE_SLAVE_IF_MODULE_n_12),
        .PENABLE_i_reg(m_apb_penable),
        .\PWDATA_i_reg[31] ({AXILITE_SLAVE_IF_MODULE_n_29,AXILITE_SLAVE_IF_MODULE_n_30,AXILITE_SLAVE_IF_MODULE_n_31,AXILITE_SLAVE_IF_MODULE_n_32,AXILITE_SLAVE_IF_MODULE_n_33,AXILITE_SLAVE_IF_MODULE_n_34,AXILITE_SLAVE_IF_MODULE_n_35,AXILITE_SLAVE_IF_MODULE_n_36,AXILITE_SLAVE_IF_MODULE_n_37,AXILITE_SLAVE_IF_MODULE_n_38,AXILITE_SLAVE_IF_MODULE_n_39,AXILITE_SLAVE_IF_MODULE_n_40,AXILITE_SLAVE_IF_MODULE_n_41,AXILITE_SLAVE_IF_MODULE_n_42,AXILITE_SLAVE_IF_MODULE_n_43,AXILITE_SLAVE_IF_MODULE_n_44,AXILITE_SLAVE_IF_MODULE_n_45,AXILITE_SLAVE_IF_MODULE_n_46,AXILITE_SLAVE_IF_MODULE_n_47,AXILITE_SLAVE_IF_MODULE_n_48,AXILITE_SLAVE_IF_MODULE_n_49,AXILITE_SLAVE_IF_MODULE_n_50,AXILITE_SLAVE_IF_MODULE_n_51,AXILITE_SLAVE_IF_MODULE_n_52,AXILITE_SLAVE_IF_MODULE_n_53,AXILITE_SLAVE_IF_MODULE_n_54,AXILITE_SLAVE_IF_MODULE_n_55,AXILITE_SLAVE_IF_MODULE_n_56,AXILITE_SLAVE_IF_MODULE_n_57,AXILITE_SLAVE_IF_MODULE_n_58,AXILITE_SLAVE_IF_MODULE_n_59,AXILITE_SLAVE_IF_MODULE_n_60}),
        .apb_rd_request(apb_rd_request),
        .apb_wr_request(apb_wr_request),
        .dphase_timeout(dphase_timeout),
        .m_apb_prdata(m_apb_prdata),
        .m_apb_pready(m_apb_pready),
        .m_apb_pslverr(m_apb_pslverr),
        .out(APB_MASTER_IF_MODULE_n_0),
        .p_0_in(p_0_in),
        .p_1_in__0(p_1_in__0),
        .s_axi_aclk(s_axi_aclk),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_aresetn(s_axi_aresetn),
        .s_axi_arready(s_axi_arready),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awready(s_axi_awready),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_bresp(\^s_axi_bresp ),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rready(s_axi_rready),
        .s_axi_rresp(\^s_axi_rresp ),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wready(s_axi_wready),
        .s_axi_wvalid(s_axi_wvalid),
        .slv_err_resp(slv_err_resp),
        .waddr_ready_sm1__0(waddr_ready_sm1__0));
  GND GND
       (.G(\<const0> ));
  axi_apb_bridge_0_multiplexor MULTIPLEXOR_MODULE
       (.PSEL_i(PSEL_i),
        .m_apb_psel(m_apb_psel),
        .p_0_in(p_0_in),
        .s_axi_aclk(s_axi_aclk));
  VCC VCC
       (.P(\<const1> ));
endmodule

(* ORIG_REF_NAME = "axilite_sif" *) 
module axi_apb_bridge_0_axilite_sif
   (s_axi_awready,
    p_0_in,
    s_axi_wready,
    apb_wr_request,
    s_axi_bvalid,
    s_axi_arready,
    apb_rd_request,
    s_axi_rvalid,
    s_axi_rresp,
    dphase_timeout,
    s_axi_bresp,
    E,
    \PADDR_i_reg[15] ,
    D,
    \PWDATA_i_reg[31] ,
    s_axi_rdata,
    s_axi_aclk,
    s_axi_aresetn,
    p_1_in__0,
    s_axi_wvalid,
    s_axi_awvalid,
    s_axi_arvalid,
    m_apb_pready,
    out,
    waddr_ready_sm1__0,
    s_axi_awaddr,
    s_axi_araddr,
    s_axi_rready,
    m_apb_pslverr,
    s_axi_bready,
    PENABLE_i_reg,
    slv_err_resp,
    s_axi_wdata,
    m_apb_prdata);
  output s_axi_awready;
  output p_0_in;
  output s_axi_wready;
  output apb_wr_request;
  output s_axi_bvalid;
  output s_axi_arready;
  output apb_rd_request;
  output s_axi_rvalid;
  output [0:0]s_axi_rresp;
  output dphase_timeout;
  output [0:0]s_axi_bresp;
  output [0:0]E;
  output \PADDR_i_reg[15] ;
  output [15:0]D;
  output [31:0]\PWDATA_i_reg[31] ;
  output [31:0]s_axi_rdata;
  input s_axi_aclk;
  input s_axi_aresetn;
  input p_1_in__0;
  input s_axi_wvalid;
  input s_axi_awvalid;
  input s_axi_arvalid;
  input [0:0]m_apb_pready;
  input [0:0]out;
  input waddr_ready_sm1__0;
  input [15:0]s_axi_awaddr;
  input [15:0]s_axi_araddr;
  input s_axi_rready;
  input [0:0]m_apb_pslverr;
  input s_axi_bready;
  input PENABLE_i_reg;
  input slv_err_resp;
  input [31:0]s_axi_wdata;
  input [31:0]m_apb_prdata;

  wire BRESP_1_i_i_1_n_0;
  wire BVALID_sm;
  wire [15:0]D;
  wire \DATA_PHASE_WDT.I_DPTO_COUNTER_n_0 ;
  wire [0:0]E;
  wire \FSM_sequential_axi_wr_rd_cs[0]_i_1_n_0 ;
  wire \FSM_sequential_axi_wr_rd_cs[0]_i_2_n_0 ;
  wire \FSM_sequential_axi_wr_rd_cs[0]_i_3_n_0 ;
  wire \FSM_sequential_axi_wr_rd_cs[1]_i_1_n_0 ;
  wire \FSM_sequential_axi_wr_rd_cs[2]_i_1_n_0 ;
  wire \FSM_sequential_axi_wr_rd_cs[2]_i_6_n_0 ;
  wire \FSM_sequential_axi_wr_rd_cs[2]_i_7_n_0 ;
  wire \FSM_sequential_axi_wr_rd_cs[2]_i_8_n_0 ;
  wire \FSM_sequential_axi_wr_rd_cs[2]_i_9_n_0 ;
  wire \FSM_sequential_axi_wr_rd_cs_reg[2]_i_3_n_0 ;
  wire \FSM_sequential_axi_wr_rd_cs_reg[2]_i_4_n_0 ;
  wire \FSM_sequential_axi_wr_rd_cs_reg[2]_i_5_n_0 ;
  wire \PADDR_i_reg[15] ;
  wire PENABLE_i_reg;
  wire [31:0]\PWDATA_i_reg[31] ;
  wire RRESP_1_i;
  wire RVALID_sm;
  wire \S_AXI_RDATA[0]_i_1_n_0 ;
  wire \S_AXI_RDATA[10]_i_1_n_0 ;
  wire \S_AXI_RDATA[11]_i_1_n_0 ;
  wire \S_AXI_RDATA[12]_i_1_n_0 ;
  wire \S_AXI_RDATA[13]_i_1_n_0 ;
  wire \S_AXI_RDATA[14]_i_1_n_0 ;
  wire \S_AXI_RDATA[15]_i_1_n_0 ;
  wire \S_AXI_RDATA[16]_i_1_n_0 ;
  wire \S_AXI_RDATA[17]_i_1_n_0 ;
  wire \S_AXI_RDATA[18]_i_1_n_0 ;
  wire \S_AXI_RDATA[19]_i_1_n_0 ;
  wire \S_AXI_RDATA[1]_i_1_n_0 ;
  wire \S_AXI_RDATA[20]_i_1_n_0 ;
  wire \S_AXI_RDATA[21]_i_1_n_0 ;
  wire \S_AXI_RDATA[22]_i_1_n_0 ;
  wire \S_AXI_RDATA[23]_i_1_n_0 ;
  wire \S_AXI_RDATA[24]_i_1_n_0 ;
  wire \S_AXI_RDATA[25]_i_1_n_0 ;
  wire \S_AXI_RDATA[26]_i_1_n_0 ;
  wire \S_AXI_RDATA[27]_i_1_n_0 ;
  wire \S_AXI_RDATA[28]_i_1_n_0 ;
  wire \S_AXI_RDATA[29]_i_1_n_0 ;
  wire \S_AXI_RDATA[2]_i_1_n_0 ;
  wire \S_AXI_RDATA[30]_i_1_n_0 ;
  wire \S_AXI_RDATA[31]_i_1_n_0 ;
  wire \S_AXI_RDATA[31]_i_2_n_0 ;
  wire \S_AXI_RDATA[3]_i_1_n_0 ;
  wire \S_AXI_RDATA[4]_i_1_n_0 ;
  wire \S_AXI_RDATA[5]_i_1_n_0 ;
  wire \S_AXI_RDATA[6]_i_1_n_0 ;
  wire \S_AXI_RDATA[7]_i_1_n_0 ;
  wire \S_AXI_RDATA[8]_i_1_n_0 ;
  wire \S_AXI_RDATA[9]_i_1_n_0 ;
  wire WREADY_i_i_2_n_0;
  wire [15:0]address_i;
  wire \address_i[0]_i_1_n_0 ;
  wire \address_i[10]_i_1_n_0 ;
  wire \address_i[11]_i_1_n_0 ;
  wire \address_i[12]_i_1_n_0 ;
  wire \address_i[13]_i_1_n_0 ;
  wire \address_i[14]_i_1_n_0 ;
  wire \address_i[15]_i_1_n_0 ;
  wire \address_i[15]_i_2_n_0 ;
  wire \address_i[1]_i_1_n_0 ;
  wire \address_i[2]_i_1_n_0 ;
  wire \address_i[3]_i_1_n_0 ;
  wire \address_i[4]_i_1_n_0 ;
  wire \address_i[5]_i_1_n_0 ;
  wire \address_i[6]_i_1_n_0 ;
  wire \address_i[7]_i_1_n_0 ;
  wire \address_i[8]_i_1_n_0 ;
  wire \address_i[9]_i_1_n_0 ;
  wire apb_rd_request;
  wire apb_wr_request;
  (* RTL_KEEP = "yes" *) wire [2:0]axi_wr_rd_cs;
  wire [2:2]axi_wr_rd_ns;
  wire dphase_timeout;
  wire [31:0]m_apb_prdata;
  wire [0:0]m_apb_pready;
  wire [0:0]m_apb_pslverr;
  wire [0:0]out;
  wire p_0_in;
  wire p_1_in__0;
  wire s_axi_aclk;
  wire [15:0]s_axi_araddr;
  wire s_axi_aresetn;
  wire s_axi_arready;
  wire s_axi_arvalid;
  wire [15:0]s_axi_awaddr;
  wire s_axi_awready;
  wire s_axi_awvalid;
  wire s_axi_bready;
  wire [0:0]s_axi_bresp;
  wire s_axi_bvalid;
  wire [31:0]s_axi_rdata;
  wire s_axi_rready;
  wire [0:0]s_axi_rresp;
  wire s_axi_rvalid;
  wire [31:0]s_axi_wdata;
  wire s_axi_wready;
  wire s_axi_wvalid;
  wire send_rd__2;
  wire send_wr_resp__1;
  wire slv_err_resp;
  wire waddr_ready_sm;
  wire waddr_ready_sm1__0;

  LUT4 #(
    .INIT(16'h0010)) 
    ARREADY_i_i_1
       (.I0(axi_wr_rd_cs[0]),
        .I1(axi_wr_rd_cs[2]),
        .I2(s_axi_arvalid),
        .I3(axi_wr_rd_cs[1]),
        .O(apb_rd_request));
  FDRE ARREADY_i_reg
       (.C(s_axi_aclk),
        .CE(1'b1),
        .D(apb_rd_request),
        .Q(s_axi_arready),
        .R(p_0_in));
  LUT1 #(
    .INIT(2'h1)) 
    AWREADY_i_i_1
       (.I0(s_axi_aresetn),
        .O(p_0_in));
  LUT6 #(
    .INIT(64'h000000008080000A)) 
    AWREADY_i_i_2
       (.I0(s_axi_awvalid),
        .I1(s_axi_rready),
        .I2(axi_wr_rd_cs[0]),
        .I3(s_axi_arvalid),
        .I4(axi_wr_rd_cs[1]),
        .I5(axi_wr_rd_cs[2]),
        .O(waddr_ready_sm));
  FDRE AWREADY_i_reg
       (.C(s_axi_aclk),
        .CE(1'b1),
        .D(waddr_ready_sm),
        .Q(s_axi_awready),
        .R(p_0_in));
  LUT4 #(
    .INIT(16'h8B88)) 
    BRESP_1_i_i_1
       (.I0(slv_err_resp),
        .I1(send_wr_resp__1),
        .I2(s_axi_bready),
        .I3(s_axi_bresp),
        .O(BRESP_1_i_i_1_n_0));
  LUT6 #(
    .INIT(64'h2222200000000000)) 
    BRESP_1_i_i_3
       (.I0(axi_wr_rd_cs[2]),
        .I1(axi_wr_rd_cs[0]),
        .I2(PENABLE_i_reg),
        .I3(m_apb_pready),
        .I4(dphase_timeout),
        .I5(axi_wr_rd_cs[1]),
        .O(send_wr_resp__1));
  FDRE BRESP_1_i_reg
       (.C(s_axi_aclk),
        .CE(1'b1),
        .D(BRESP_1_i_i_1_n_0),
        .Q(s_axi_bresp),
        .R(p_0_in));
  LUT5 #(
    .INIT(32'h00808880)) 
    BVALID_i_i_1
       (.I0(axi_wr_rd_cs[2]),
        .I1(axi_wr_rd_cs[1]),
        .I2(p_1_in__0),
        .I3(axi_wr_rd_cs[0]),
        .I4(s_axi_bready),
        .O(BVALID_sm));
  FDRE BVALID_i_reg
       (.C(s_axi_aclk),
        .CE(1'b1),
        .D(BVALID_sm),
        .Q(s_axi_bvalid),
        .R(p_0_in));
  axi_apb_bridge_0_counter_f \DATA_PHASE_WDT.I_DPTO_COUNTER 
       (.\DATA_PHASE_WDT.data_timeout_reg (\DATA_PHASE_WDT.I_DPTO_COUNTER_n_0 ),
        .\DATA_PHASE_WDT.data_timeout_reg_0 (dphase_timeout),
        .\FSM_sequential_axi_wr_rd_cs_reg[1] (WREADY_i_i_2_n_0),
        .PENABLE_i_reg(PENABLE_i_reg),
        .m_apb_pready(m_apb_pready),
        .out(axi_wr_rd_cs),
        .p_1_in__0(p_1_in__0),
        .s_axi_aclk(s_axi_aclk),
        .s_axi_aresetn(s_axi_aresetn),
        .s_axi_arvalid(s_axi_arvalid),
        .waddr_ready_sm1__0(waddr_ready_sm1__0));
  FDRE \DATA_PHASE_WDT.data_timeout_reg 
       (.C(s_axi_aclk),
        .CE(1'b1),
        .D(\DATA_PHASE_WDT.I_DPTO_COUNTER_n_0 ),
        .Q(dphase_timeout),
        .R(p_0_in));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \FSM_sequential_axi_wr_rd_cs[0]_i_1 
       (.I0(\FSM_sequential_axi_wr_rd_cs[0]_i_2_n_0 ),
        .I1(axi_wr_rd_cs[0]),
        .I2(\FSM_sequential_axi_wr_rd_cs[0]_i_3_n_0 ),
        .I3(\FSM_sequential_axi_wr_rd_cs_reg[2]_i_3_n_0 ),
        .I4(axi_wr_rd_cs[0]),
        .O(\FSM_sequential_axi_wr_rd_cs[0]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'h0400)) 
    \FSM_sequential_axi_wr_rd_cs[0]_i_2 
       (.I0(axi_wr_rd_cs[2]),
        .I1(axi_wr_rd_cs[1]),
        .I2(s_axi_wvalid),
        .I3(s_axi_awvalid),
        .O(\FSM_sequential_axi_wr_rd_cs[0]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hCCCCCCCCBBBB8BBB)) 
    \FSM_sequential_axi_wr_rd_cs[0]_i_3 
       (.I0(p_1_in__0),
        .I1(axi_wr_rd_cs[1]),
        .I2(s_axi_wvalid),
        .I3(s_axi_awvalid),
        .I4(s_axi_arvalid),
        .I5(axi_wr_rd_cs[2]),
        .O(\FSM_sequential_axi_wr_rd_cs[0]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h0FEAFFFF0FEA0000)) 
    \FSM_sequential_axi_wr_rd_cs[1]_i_1 
       (.I0(axi_wr_rd_cs[2]),
        .I1(p_1_in__0),
        .I2(axi_wr_rd_cs[1]),
        .I3(axi_wr_rd_cs[0]),
        .I4(\FSM_sequential_axi_wr_rd_cs_reg[2]_i_3_n_0 ),
        .I5(axi_wr_rd_cs[1]),
        .O(\FSM_sequential_axi_wr_rd_cs[1]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hB8)) 
    \FSM_sequential_axi_wr_rd_cs[2]_i_1 
       (.I0(axi_wr_rd_ns),
        .I1(\FSM_sequential_axi_wr_rd_cs_reg[2]_i_3_n_0 ),
        .I2(axi_wr_rd_cs[2]),
        .O(\FSM_sequential_axi_wr_rd_cs[2]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h5AF150F1)) 
    \FSM_sequential_axi_wr_rd_cs[2]_i_2 
       (.I0(axi_wr_rd_cs[1]),
        .I1(s_axi_arvalid),
        .I2(axi_wr_rd_cs[2]),
        .I3(axi_wr_rd_cs[0]),
        .I4(s_axi_awvalid),
        .O(axi_wr_rd_ns));
  LUT3 #(
    .INIT(8'hFE)) 
    \FSM_sequential_axi_wr_rd_cs[2]_i_6 
       (.I0(axi_wr_rd_cs[0]),
        .I1(s_axi_awvalid),
        .I2(s_axi_arvalid),
        .O(\FSM_sequential_axi_wr_rd_cs[2]_i_6_n_0 ));
  LUT5 #(
    .INIT(32'hBBBBB888)) 
    \FSM_sequential_axi_wr_rd_cs[2]_i_7 
       (.I0(s_axi_rready),
        .I1(axi_wr_rd_cs[0]),
        .I2(PENABLE_i_reg),
        .I3(m_apb_pready),
        .I4(dphase_timeout),
        .O(\FSM_sequential_axi_wr_rd_cs[2]_i_7_n_0 ));
  LUT2 #(
    .INIT(4'hB)) 
    \FSM_sequential_axi_wr_rd_cs[2]_i_8 
       (.I0(s_axi_wvalid),
        .I1(axi_wr_rd_cs[0]),
        .O(\FSM_sequential_axi_wr_rd_cs[2]_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hBBBBB888)) 
    \FSM_sequential_axi_wr_rd_cs[2]_i_9 
       (.I0(s_axi_bready),
        .I1(axi_wr_rd_cs[0]),
        .I2(PENABLE_i_reg),
        .I3(m_apb_pready),
        .I4(dphase_timeout),
        .O(\FSM_sequential_axi_wr_rd_cs[2]_i_9_n_0 ));
  (* FSM_ENCODED_STATES = "write:110,wr_resp:111,read:010,read_wait:001,rd_resp:011,write_wait:100,axi_idle:000,write_w_wait:101" *) 
  (* KEEP = "yes" *) 
  FDRE \FSM_sequential_axi_wr_rd_cs_reg[0] 
       (.C(s_axi_aclk),
        .CE(1'b1),
        .D(\FSM_sequential_axi_wr_rd_cs[0]_i_1_n_0 ),
        .Q(axi_wr_rd_cs[0]),
        .R(p_0_in));
  (* FSM_ENCODED_STATES = "write:110,wr_resp:111,read:010,read_wait:001,rd_resp:011,write_wait:100,axi_idle:000,write_w_wait:101" *) 
  (* KEEP = "yes" *) 
  FDRE \FSM_sequential_axi_wr_rd_cs_reg[1] 
       (.C(s_axi_aclk),
        .CE(1'b1),
        .D(\FSM_sequential_axi_wr_rd_cs[1]_i_1_n_0 ),
        .Q(axi_wr_rd_cs[1]),
        .R(p_0_in));
  (* FSM_ENCODED_STATES = "write:110,wr_resp:111,read:010,read_wait:001,rd_resp:011,write_wait:100,axi_idle:000,write_w_wait:101" *) 
  (* KEEP = "yes" *) 
  FDRE \FSM_sequential_axi_wr_rd_cs_reg[2] 
       (.C(s_axi_aclk),
        .CE(1'b1),
        .D(\FSM_sequential_axi_wr_rd_cs[2]_i_1_n_0 ),
        .Q(axi_wr_rd_cs[2]),
        .R(p_0_in));
  MUXF8 \FSM_sequential_axi_wr_rd_cs_reg[2]_i_3 
       (.I0(\FSM_sequential_axi_wr_rd_cs_reg[2]_i_4_n_0 ),
        .I1(\FSM_sequential_axi_wr_rd_cs_reg[2]_i_5_n_0 ),
        .O(\FSM_sequential_axi_wr_rd_cs_reg[2]_i_3_n_0 ),
        .S(axi_wr_rd_cs[2]));
  MUXF7 \FSM_sequential_axi_wr_rd_cs_reg[2]_i_4 
       (.I0(\FSM_sequential_axi_wr_rd_cs[2]_i_6_n_0 ),
        .I1(\FSM_sequential_axi_wr_rd_cs[2]_i_7_n_0 ),
        .O(\FSM_sequential_axi_wr_rd_cs_reg[2]_i_4_n_0 ),
        .S(axi_wr_rd_cs[1]));
  MUXF7 \FSM_sequential_axi_wr_rd_cs_reg[2]_i_5 
       (.I0(\FSM_sequential_axi_wr_rd_cs[2]_i_8_n_0 ),
        .I1(\FSM_sequential_axi_wr_rd_cs[2]_i_9_n_0 ),
        .O(\FSM_sequential_axi_wr_rd_cs_reg[2]_i_5_n_0 ),
        .S(axi_wr_rd_cs[1]));
  LUT6 #(
    .INIT(64'hEFEAFFFF45400000)) 
    \PADDR_i[0]_i_1 
       (.I0(apb_rd_request),
        .I1(s_axi_awaddr[0]),
        .I2(waddr_ready_sm),
        .I3(address_i[0]),
        .I4(apb_wr_request),
        .I5(s_axi_araddr[0]),
        .O(D[0]));
  LUT6 #(
    .INIT(64'hEFEAFFFF45400000)) 
    \PADDR_i[10]_i_1 
       (.I0(apb_rd_request),
        .I1(s_axi_awaddr[10]),
        .I2(waddr_ready_sm),
        .I3(address_i[10]),
        .I4(apb_wr_request),
        .I5(s_axi_araddr[10]),
        .O(D[10]));
  LUT6 #(
    .INIT(64'hEFEAFFFF45400000)) 
    \PADDR_i[11]_i_1 
       (.I0(apb_rd_request),
        .I1(s_axi_awaddr[11]),
        .I2(waddr_ready_sm),
        .I3(address_i[11]),
        .I4(apb_wr_request),
        .I5(s_axi_araddr[11]),
        .O(D[11]));
  LUT6 #(
    .INIT(64'hEFEAFFFF45400000)) 
    \PADDR_i[12]_i_1 
       (.I0(apb_rd_request),
        .I1(s_axi_awaddr[12]),
        .I2(waddr_ready_sm),
        .I3(address_i[12]),
        .I4(apb_wr_request),
        .I5(s_axi_araddr[12]),
        .O(D[12]));
  LUT6 #(
    .INIT(64'hEFEAFFFF45400000)) 
    \PADDR_i[13]_i_1 
       (.I0(apb_rd_request),
        .I1(s_axi_awaddr[13]),
        .I2(waddr_ready_sm),
        .I3(address_i[13]),
        .I4(apb_wr_request),
        .I5(s_axi_araddr[13]),
        .O(D[13]));
  LUT6 #(
    .INIT(64'hEFEAFFFF45400000)) 
    \PADDR_i[14]_i_1 
       (.I0(apb_rd_request),
        .I1(s_axi_awaddr[14]),
        .I2(waddr_ready_sm),
        .I3(address_i[14]),
        .I4(apb_wr_request),
        .I5(s_axi_araddr[14]),
        .O(D[14]));
  LUT6 #(
    .INIT(64'hCCCCCCCC00000F0A)) 
    \PADDR_i[15]_i_1 
       (.I0(waddr_ready_sm1__0),
        .I1(WREADY_i_i_2_n_0),
        .I2(axi_wr_rd_cs[1]),
        .I3(s_axi_arvalid),
        .I4(axi_wr_rd_cs[2]),
        .I5(axi_wr_rd_cs[0]),
        .O(\PADDR_i_reg[15] ));
  LUT6 #(
    .INIT(64'hEFEAFFFF45400000)) 
    \PADDR_i[15]_i_2 
       (.I0(apb_rd_request),
        .I1(s_axi_awaddr[15]),
        .I2(waddr_ready_sm),
        .I3(address_i[15]),
        .I4(apb_wr_request),
        .I5(s_axi_araddr[15]),
        .O(D[15]));
  LUT6 #(
    .INIT(64'hEFEAFFFF45400000)) 
    \PADDR_i[1]_i_1 
       (.I0(apb_rd_request),
        .I1(s_axi_awaddr[1]),
        .I2(waddr_ready_sm),
        .I3(address_i[1]),
        .I4(apb_wr_request),
        .I5(s_axi_araddr[1]),
        .O(D[1]));
  LUT6 #(
    .INIT(64'hEFEAFFFF45400000)) 
    \PADDR_i[2]_i_1 
       (.I0(apb_rd_request),
        .I1(s_axi_awaddr[2]),
        .I2(waddr_ready_sm),
        .I3(address_i[2]),
        .I4(apb_wr_request),
        .I5(s_axi_araddr[2]),
        .O(D[2]));
  LUT6 #(
    .INIT(64'hEFEAFFFF45400000)) 
    \PADDR_i[3]_i_1 
       (.I0(apb_rd_request),
        .I1(s_axi_awaddr[3]),
        .I2(waddr_ready_sm),
        .I3(address_i[3]),
        .I4(apb_wr_request),
        .I5(s_axi_araddr[3]),
        .O(D[3]));
  LUT6 #(
    .INIT(64'hEFEAFFFF45400000)) 
    \PADDR_i[4]_i_1 
       (.I0(apb_rd_request),
        .I1(s_axi_awaddr[4]),
        .I2(waddr_ready_sm),
        .I3(address_i[4]),
        .I4(apb_wr_request),
        .I5(s_axi_araddr[4]),
        .O(D[4]));
  LUT6 #(
    .INIT(64'hEFEAFFFF45400000)) 
    \PADDR_i[5]_i_1 
       (.I0(apb_rd_request),
        .I1(s_axi_awaddr[5]),
        .I2(waddr_ready_sm),
        .I3(address_i[5]),
        .I4(apb_wr_request),
        .I5(s_axi_araddr[5]),
        .O(D[5]));
  LUT6 #(
    .INIT(64'hEFEAFFFF45400000)) 
    \PADDR_i[6]_i_1 
       (.I0(apb_rd_request),
        .I1(s_axi_awaddr[6]),
        .I2(waddr_ready_sm),
        .I3(address_i[6]),
        .I4(apb_wr_request),
        .I5(s_axi_araddr[6]),
        .O(D[6]));
  LUT6 #(
    .INIT(64'hEFEAFFFF45400000)) 
    \PADDR_i[7]_i_1 
       (.I0(apb_rd_request),
        .I1(s_axi_awaddr[7]),
        .I2(waddr_ready_sm),
        .I3(address_i[7]),
        .I4(apb_wr_request),
        .I5(s_axi_araddr[7]),
        .O(D[7]));
  LUT6 #(
    .INIT(64'hEFEAFFFF45400000)) 
    \PADDR_i[8]_i_1 
       (.I0(apb_rd_request),
        .I1(s_axi_awaddr[8]),
        .I2(waddr_ready_sm),
        .I3(address_i[8]),
        .I4(apb_wr_request),
        .I5(s_axi_araddr[8]),
        .O(D[8]));
  LUT6 #(
    .INIT(64'hEFEAFFFF45400000)) 
    \PADDR_i[9]_i_1 
       (.I0(apb_rd_request),
        .I1(s_axi_awaddr[9]),
        .I2(waddr_ready_sm),
        .I3(address_i[9]),
        .I4(apb_wr_request),
        .I5(s_axi_araddr[9]),
        .O(D[9]));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[0]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[0]),
        .O(\PWDATA_i_reg[31] [0]));
  (* SOFT_HLUTNM = "soft_lutpair36" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[10]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[10]),
        .O(\PWDATA_i_reg[31] [10]));
  (* SOFT_HLUTNM = "soft_lutpair37" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[11]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[11]),
        .O(\PWDATA_i_reg[31] [11]));
  (* SOFT_HLUTNM = "soft_lutpair38" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[12]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[12]),
        .O(\PWDATA_i_reg[31] [12]));
  (* SOFT_HLUTNM = "soft_lutpair39" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[13]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[13]),
        .O(\PWDATA_i_reg[31] [13]));
  (* SOFT_HLUTNM = "soft_lutpair40" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[14]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[14]),
        .O(\PWDATA_i_reg[31] [14]));
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[15]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[15]),
        .O(\PWDATA_i_reg[31] [15]));
  (* SOFT_HLUTNM = "soft_lutpair40" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[16]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[16]),
        .O(\PWDATA_i_reg[31] [16]));
  (* SOFT_HLUTNM = "soft_lutpair39" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[17]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[17]),
        .O(\PWDATA_i_reg[31] [17]));
  (* SOFT_HLUTNM = "soft_lutpair38" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[18]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[18]),
        .O(\PWDATA_i_reg[31] [18]));
  (* SOFT_HLUTNM = "soft_lutpair37" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[19]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[19]),
        .O(\PWDATA_i_reg[31] [19]));
  (* SOFT_HLUTNM = "soft_lutpair27" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[1]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[1]),
        .O(\PWDATA_i_reg[31] [1]));
  (* SOFT_HLUTNM = "soft_lutpair36" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[20]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[20]),
        .O(\PWDATA_i_reg[31] [20]));
  (* SOFT_HLUTNM = "soft_lutpair35" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[21]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[21]),
        .O(\PWDATA_i_reg[31] [21]));
  (* SOFT_HLUTNM = "soft_lutpair34" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[22]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[22]),
        .O(\PWDATA_i_reg[31] [22]));
  (* SOFT_HLUTNM = "soft_lutpair33" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[23]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[23]),
        .O(\PWDATA_i_reg[31] [23]));
  (* SOFT_HLUTNM = "soft_lutpair32" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[24]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[24]),
        .O(\PWDATA_i_reg[31] [24]));
  (* SOFT_HLUTNM = "soft_lutpair31" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[25]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[25]),
        .O(\PWDATA_i_reg[31] [25]));
  (* SOFT_HLUTNM = "soft_lutpair30" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[26]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[26]),
        .O(\PWDATA_i_reg[31] [26]));
  (* SOFT_HLUTNM = "soft_lutpair29" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[27]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[27]),
        .O(\PWDATA_i_reg[31] [27]));
  (* SOFT_HLUTNM = "soft_lutpair28" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[28]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[28]),
        .O(\PWDATA_i_reg[31] [28]));
  (* SOFT_HLUTNM = "soft_lutpair27" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[29]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[29]),
        .O(\PWDATA_i_reg[31] [29]));
  (* SOFT_HLUTNM = "soft_lutpair28" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[2]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[2]),
        .O(\PWDATA_i_reg[31] [2]));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[30]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[30]),
        .O(\PWDATA_i_reg[31] [30]));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT4 #(
    .INIT(16'hFEAA)) 
    \PWDATA_i[31]_i_1 
       (.I0(apb_wr_request),
        .I1(m_apb_pready),
        .I2(dphase_timeout),
        .I3(out),
        .O(E));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[31]_i_2 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[31]),
        .O(\PWDATA_i_reg[31] [31]));
  (* SOFT_HLUTNM = "soft_lutpair29" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[3]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[3]),
        .O(\PWDATA_i_reg[31] [3]));
  (* SOFT_HLUTNM = "soft_lutpair30" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[4]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[4]),
        .O(\PWDATA_i_reg[31] [4]));
  (* SOFT_HLUTNM = "soft_lutpair31" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[5]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[5]),
        .O(\PWDATA_i_reg[31] [5]));
  (* SOFT_HLUTNM = "soft_lutpair32" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[6]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[6]),
        .O(\PWDATA_i_reg[31] [6]));
  (* SOFT_HLUTNM = "soft_lutpair33" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[7]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[7]),
        .O(\PWDATA_i_reg[31] [7]));
  (* SOFT_HLUTNM = "soft_lutpair34" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[8]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[8]),
        .O(\PWDATA_i_reg[31] [8]));
  (* SOFT_HLUTNM = "soft_lutpair35" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \PWDATA_i[9]_i_1 
       (.I0(apb_wr_request),
        .I1(s_axi_wdata[9]),
        .O(\PWDATA_i_reg[31] [9]));
  LUT5 #(
    .INIT(32'h8A800000)) 
    RRESP_1_i_i_1
       (.I0(send_rd__2),
        .I1(m_apb_pslverr),
        .I2(m_apb_pready),
        .I3(dphase_timeout),
        .I4(out),
        .O(RRESP_1_i));
  FDRE RRESP_1_i_reg
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(RRESP_1_i),
        .Q(s_axi_rresp),
        .R(p_0_in));
  LUT5 #(
    .INIT(32'h00404440)) 
    RVALID_i_i_1
       (.I0(axi_wr_rd_cs[2]),
        .I1(axi_wr_rd_cs[1]),
        .I2(p_1_in__0),
        .I3(axi_wr_rd_cs[0]),
        .I4(s_axi_rready),
        .O(RVALID_sm));
  FDRE RVALID_i_reg
       (.C(s_axi_aclk),
        .CE(1'b1),
        .D(RVALID_sm),
        .Q(s_axi_rvalid),
        .R(p_0_in));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[0]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[0]),
        .I3(out),
        .O(\S_AXI_RDATA[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[10]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[10]),
        .I3(out),
        .O(\S_AXI_RDATA[10]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[11]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[11]),
        .I3(out),
        .O(\S_AXI_RDATA[11]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[12]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[12]),
        .I3(out),
        .O(\S_AXI_RDATA[12]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[13]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[13]),
        .I3(out),
        .O(\S_AXI_RDATA[13]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[14]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[14]),
        .I3(out),
        .O(\S_AXI_RDATA[14]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[15]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[15]),
        .I3(out),
        .O(\S_AXI_RDATA[15]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[16]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[16]),
        .I3(out),
        .O(\S_AXI_RDATA[16]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[17]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[17]),
        .I3(out),
        .O(\S_AXI_RDATA[17]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[18]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[18]),
        .I3(out),
        .O(\S_AXI_RDATA[18]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[19]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[19]),
        .I3(out),
        .O(\S_AXI_RDATA[19]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[1]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[1]),
        .I3(out),
        .O(\S_AXI_RDATA[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[20]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[20]),
        .I3(out),
        .O(\S_AXI_RDATA[20]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[21]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[21]),
        .I3(out),
        .O(\S_AXI_RDATA[21]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[22]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[22]),
        .I3(out),
        .O(\S_AXI_RDATA[22]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[23]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[23]),
        .I3(out),
        .O(\S_AXI_RDATA[23]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[24]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[24]),
        .I3(out),
        .O(\S_AXI_RDATA[24]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[25]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[25]),
        .I3(out),
        .O(\S_AXI_RDATA[25]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[26]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[26]),
        .I3(out),
        .O(\S_AXI_RDATA[26]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[27]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[27]),
        .I3(out),
        .O(\S_AXI_RDATA[27]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[28]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[28]),
        .I3(out),
        .O(\S_AXI_RDATA[28]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[29]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[29]),
        .I3(out),
        .O(\S_AXI_RDATA[29]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[2]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[2]),
        .I3(out),
        .O(\S_AXI_RDATA[2]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[30]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[30]),
        .I3(out),
        .O(\S_AXI_RDATA[30]_i_1_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \S_AXI_RDATA[31]_i_1 
       (.I0(send_rd__2),
        .I1(s_axi_rready),
        .O(\S_AXI_RDATA[31]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[31]_i_2 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[31]),
        .I3(out),
        .O(\S_AXI_RDATA[31]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h1010101010000000)) 
    \S_AXI_RDATA[31]_i_3 
       (.I0(axi_wr_rd_cs[0]),
        .I1(axi_wr_rd_cs[2]),
        .I2(axi_wr_rd_cs[1]),
        .I3(PENABLE_i_reg),
        .I4(m_apb_pready),
        .I5(dphase_timeout),
        .O(send_rd__2));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[3]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[3]),
        .I3(out),
        .O(\S_AXI_RDATA[3]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[4]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[4]),
        .I3(out),
        .O(\S_AXI_RDATA[4]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[5]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[5]),
        .I3(out),
        .O(\S_AXI_RDATA[5]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[6]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[6]),
        .I3(out),
        .O(\S_AXI_RDATA[6]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[7]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[7]),
        .I3(out),
        .O(\S_AXI_RDATA[7]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[8]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[8]),
        .I3(out),
        .O(\S_AXI_RDATA[8]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \S_AXI_RDATA[9]_i_1 
       (.I0(send_rd__2),
        .I1(m_apb_pready),
        .I2(m_apb_prdata[9]),
        .I3(out),
        .O(\S_AXI_RDATA[9]_i_1_n_0 ));
  FDRE \S_AXI_RDATA_reg[0] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[0]_i_1_n_0 ),
        .Q(s_axi_rdata[0]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[10] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[10]_i_1_n_0 ),
        .Q(s_axi_rdata[10]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[11] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[11]_i_1_n_0 ),
        .Q(s_axi_rdata[11]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[12] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[12]_i_1_n_0 ),
        .Q(s_axi_rdata[12]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[13] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[13]_i_1_n_0 ),
        .Q(s_axi_rdata[13]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[14] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[14]_i_1_n_0 ),
        .Q(s_axi_rdata[14]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[15] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[15]_i_1_n_0 ),
        .Q(s_axi_rdata[15]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[16] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[16]_i_1_n_0 ),
        .Q(s_axi_rdata[16]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[17] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[17]_i_1_n_0 ),
        .Q(s_axi_rdata[17]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[18] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[18]_i_1_n_0 ),
        .Q(s_axi_rdata[18]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[19] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[19]_i_1_n_0 ),
        .Q(s_axi_rdata[19]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[1] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[1]_i_1_n_0 ),
        .Q(s_axi_rdata[1]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[20] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[20]_i_1_n_0 ),
        .Q(s_axi_rdata[20]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[21] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[21]_i_1_n_0 ),
        .Q(s_axi_rdata[21]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[22] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[22]_i_1_n_0 ),
        .Q(s_axi_rdata[22]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[23] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[23]_i_1_n_0 ),
        .Q(s_axi_rdata[23]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[24] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[24]_i_1_n_0 ),
        .Q(s_axi_rdata[24]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[25] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[25]_i_1_n_0 ),
        .Q(s_axi_rdata[25]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[26] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[26]_i_1_n_0 ),
        .Q(s_axi_rdata[26]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[27] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[27]_i_1_n_0 ),
        .Q(s_axi_rdata[27]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[28] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[28]_i_1_n_0 ),
        .Q(s_axi_rdata[28]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[29] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[29]_i_1_n_0 ),
        .Q(s_axi_rdata[29]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[2] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[2]_i_1_n_0 ),
        .Q(s_axi_rdata[2]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[30] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[30]_i_1_n_0 ),
        .Q(s_axi_rdata[30]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[31] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[31]_i_2_n_0 ),
        .Q(s_axi_rdata[31]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[3] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[3]_i_1_n_0 ),
        .Q(s_axi_rdata[3]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[4] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[4]_i_1_n_0 ),
        .Q(s_axi_rdata[4]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[5] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[5]_i_1_n_0 ),
        .Q(s_axi_rdata[5]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[6] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[6]_i_1_n_0 ),
        .Q(s_axi_rdata[6]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[7] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[7]_i_1_n_0 ),
        .Q(s_axi_rdata[7]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[8] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[8]_i_1_n_0 ),
        .Q(s_axi_rdata[8]),
        .R(p_0_in));
  FDRE \S_AXI_RDATA_reg[9] 
       (.C(s_axi_aclk),
        .CE(\S_AXI_RDATA[31]_i_1_n_0 ),
        .D(\S_AXI_RDATA[9]_i_1_n_0 ),
        .Q(s_axi_rdata[9]),
        .R(p_0_in));
  LUT6 #(
    .INIT(64'h8888888888888B88)) 
    WREADY_i_i_1
       (.I0(WREADY_i_i_2_n_0),
        .I1(axi_wr_rd_cs[0]),
        .I2(s_axi_arvalid),
        .I3(waddr_ready_sm1__0),
        .I4(axi_wr_rd_cs[1]),
        .I5(axi_wr_rd_cs[2]),
        .O(apb_wr_request));
  LUT5 #(
    .INIT(32'h0F800000)) 
    WREADY_i_i_2
       (.I0(s_axi_awvalid),
        .I1(s_axi_rready),
        .I2(axi_wr_rd_cs[1]),
        .I3(axi_wr_rd_cs[2]),
        .I4(s_axi_wvalid),
        .O(WREADY_i_i_2_n_0));
  FDRE WREADY_i_reg
       (.C(s_axi_aclk),
        .CE(1'b1),
        .D(apb_wr_request),
        .Q(s_axi_wready),
        .R(p_0_in));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \address_i[0]_i_1 
       (.I0(s_axi_awaddr[0]),
        .I1(s_axi_araddr[0]),
        .I2(waddr_ready_sm),
        .O(\address_i[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \address_i[10]_i_1 
       (.I0(s_axi_awaddr[10]),
        .I1(s_axi_araddr[10]),
        .I2(waddr_ready_sm),
        .O(\address_i[10]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \address_i[11]_i_1 
       (.I0(s_axi_awaddr[11]),
        .I1(s_axi_araddr[11]),
        .I2(waddr_ready_sm),
        .O(\address_i[11]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \address_i[12]_i_1 
       (.I0(s_axi_awaddr[12]),
        .I1(s_axi_araddr[12]),
        .I2(waddr_ready_sm),
        .O(\address_i[12]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \address_i[13]_i_1 
       (.I0(s_axi_awaddr[13]),
        .I1(s_axi_araddr[13]),
        .I2(waddr_ready_sm),
        .O(\address_i[13]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \address_i[14]_i_1 
       (.I0(s_axi_awaddr[14]),
        .I1(s_axi_araddr[14]),
        .I2(waddr_ready_sm),
        .O(\address_i[14]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hAAAAAABA)) 
    \address_i[15]_i_1 
       (.I0(waddr_ready_sm),
        .I1(axi_wr_rd_cs[1]),
        .I2(s_axi_arvalid),
        .I3(axi_wr_rd_cs[2]),
        .I4(axi_wr_rd_cs[0]),
        .O(\address_i[15]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \address_i[15]_i_2 
       (.I0(s_axi_awaddr[15]),
        .I1(s_axi_araddr[15]),
        .I2(waddr_ready_sm),
        .O(\address_i[15]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \address_i[1]_i_1 
       (.I0(s_axi_awaddr[1]),
        .I1(s_axi_araddr[1]),
        .I2(waddr_ready_sm),
        .O(\address_i[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \address_i[2]_i_1 
       (.I0(s_axi_awaddr[2]),
        .I1(s_axi_araddr[2]),
        .I2(waddr_ready_sm),
        .O(\address_i[2]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \address_i[3]_i_1 
       (.I0(s_axi_awaddr[3]),
        .I1(s_axi_araddr[3]),
        .I2(waddr_ready_sm),
        .O(\address_i[3]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \address_i[4]_i_1 
       (.I0(s_axi_awaddr[4]),
        .I1(s_axi_araddr[4]),
        .I2(waddr_ready_sm),
        .O(\address_i[4]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \address_i[5]_i_1 
       (.I0(s_axi_awaddr[5]),
        .I1(s_axi_araddr[5]),
        .I2(waddr_ready_sm),
        .O(\address_i[5]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \address_i[6]_i_1 
       (.I0(s_axi_awaddr[6]),
        .I1(s_axi_araddr[6]),
        .I2(waddr_ready_sm),
        .O(\address_i[6]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \address_i[7]_i_1 
       (.I0(s_axi_awaddr[7]),
        .I1(s_axi_araddr[7]),
        .I2(waddr_ready_sm),
        .O(\address_i[7]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \address_i[8]_i_1 
       (.I0(s_axi_awaddr[8]),
        .I1(s_axi_araddr[8]),
        .I2(waddr_ready_sm),
        .O(\address_i[8]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT3 #(
    .INIT(8'hAC)) 
    \address_i[9]_i_1 
       (.I0(s_axi_awaddr[9]),
        .I1(s_axi_araddr[9]),
        .I2(waddr_ready_sm),
        .O(\address_i[9]_i_1_n_0 ));
  FDRE \address_i_reg[0] 
       (.C(s_axi_aclk),
        .CE(\address_i[15]_i_1_n_0 ),
        .D(\address_i[0]_i_1_n_0 ),
        .Q(address_i[0]),
        .R(p_0_in));
  FDRE \address_i_reg[10] 
       (.C(s_axi_aclk),
        .CE(\address_i[15]_i_1_n_0 ),
        .D(\address_i[10]_i_1_n_0 ),
        .Q(address_i[10]),
        .R(p_0_in));
  FDRE \address_i_reg[11] 
       (.C(s_axi_aclk),
        .CE(\address_i[15]_i_1_n_0 ),
        .D(\address_i[11]_i_1_n_0 ),
        .Q(address_i[11]),
        .R(p_0_in));
  FDRE \address_i_reg[12] 
       (.C(s_axi_aclk),
        .CE(\address_i[15]_i_1_n_0 ),
        .D(\address_i[12]_i_1_n_0 ),
        .Q(address_i[12]),
        .R(p_0_in));
  FDRE \address_i_reg[13] 
       (.C(s_axi_aclk),
        .CE(\address_i[15]_i_1_n_0 ),
        .D(\address_i[13]_i_1_n_0 ),
        .Q(address_i[13]),
        .R(p_0_in));
  FDRE \address_i_reg[14] 
       (.C(s_axi_aclk),
        .CE(\address_i[15]_i_1_n_0 ),
        .D(\address_i[14]_i_1_n_0 ),
        .Q(address_i[14]),
        .R(p_0_in));
  FDRE \address_i_reg[15] 
       (.C(s_axi_aclk),
        .CE(\address_i[15]_i_1_n_0 ),
        .D(\address_i[15]_i_2_n_0 ),
        .Q(address_i[15]),
        .R(p_0_in));
  FDRE \address_i_reg[1] 
       (.C(s_axi_aclk),
        .CE(\address_i[15]_i_1_n_0 ),
        .D(\address_i[1]_i_1_n_0 ),
        .Q(address_i[1]),
        .R(p_0_in));
  FDRE \address_i_reg[2] 
       (.C(s_axi_aclk),
        .CE(\address_i[15]_i_1_n_0 ),
        .D(\address_i[2]_i_1_n_0 ),
        .Q(address_i[2]),
        .R(p_0_in));
  FDRE \address_i_reg[3] 
       (.C(s_axi_aclk),
        .CE(\address_i[15]_i_1_n_0 ),
        .D(\address_i[3]_i_1_n_0 ),
        .Q(address_i[3]),
        .R(p_0_in));
  FDRE \address_i_reg[4] 
       (.C(s_axi_aclk),
        .CE(\address_i[15]_i_1_n_0 ),
        .D(\address_i[4]_i_1_n_0 ),
        .Q(address_i[4]),
        .R(p_0_in));
  FDRE \address_i_reg[5] 
       (.C(s_axi_aclk),
        .CE(\address_i[15]_i_1_n_0 ),
        .D(\address_i[5]_i_1_n_0 ),
        .Q(address_i[5]),
        .R(p_0_in));
  FDRE \address_i_reg[6] 
       (.C(s_axi_aclk),
        .CE(\address_i[15]_i_1_n_0 ),
        .D(\address_i[6]_i_1_n_0 ),
        .Q(address_i[6]),
        .R(p_0_in));
  FDRE \address_i_reg[7] 
       (.C(s_axi_aclk),
        .CE(\address_i[15]_i_1_n_0 ),
        .D(\address_i[7]_i_1_n_0 ),
        .Q(address_i[7]),
        .R(p_0_in));
  FDRE \address_i_reg[8] 
       (.C(s_axi_aclk),
        .CE(\address_i[15]_i_1_n_0 ),
        .D(\address_i[8]_i_1_n_0 ),
        .Q(address_i[8]),
        .R(p_0_in));
  FDRE \address_i_reg[9] 
       (.C(s_axi_aclk),
        .CE(\address_i[15]_i_1_n_0 ),
        .D(\address_i[9]_i_1_n_0 ),
        .Q(address_i[9]),
        .R(p_0_in));
endmodule

(* ORIG_REF_NAME = "counter_f" *) 
module axi_apb_bridge_0_counter_f
   (\DATA_PHASE_WDT.data_timeout_reg ,
    s_axi_aclk,
    s_axi_aresetn,
    out,
    p_1_in__0,
    \FSM_sequential_axi_wr_rd_cs_reg[1] ,
    waddr_ready_sm1__0,
    s_axi_arvalid,
    \DATA_PHASE_WDT.data_timeout_reg_0 ,
    m_apb_pready,
    PENABLE_i_reg);
  output \DATA_PHASE_WDT.data_timeout_reg ;
  input s_axi_aclk;
  input s_axi_aresetn;
  input [2:0]out;
  input p_1_in__0;
  input \FSM_sequential_axi_wr_rd_cs_reg[1] ;
  input waddr_ready_sm1__0;
  input s_axi_arvalid;
  input \DATA_PHASE_WDT.data_timeout_reg_0 ;
  input [0:0]m_apb_pready;
  input PENABLE_i_reg;

  wire \DATA_PHASE_WDT.data_timeout_reg ;
  wire \DATA_PHASE_WDT.data_timeout_reg_0 ;
  wire \FSM_sequential_axi_wr_rd_cs_reg[1] ;
  wire PENABLE_i_reg;
  wire cntr_enable__2;
  wire \icount_out[0]_i_1_n_0 ;
  wire \icount_out[1]_i_1_n_0 ;
  wire \icount_out[2]_i_1_n_0 ;
  wire \icount_out[3]_i_1_n_0 ;
  wire \icount_out[3]_i_2_n_0 ;
  wire \icount_out[3]_i_3_n_0 ;
  wire \icount_out[4]_i_1_n_0 ;
  wire \icount_out[4]_i_2_n_0 ;
  wire \icount_out_reg_n_0_[0] ;
  wire \icount_out_reg_n_0_[1] ;
  wire \icount_out_reg_n_0_[2] ;
  wire \icount_out_reg_n_0_[3] ;
  wire load_cntr__0;
  wire [0:0]m_apb_pready;
  wire [2:0]out;
  wire p_1_in__0;
  wire s_axi_aclk;
  wire s_axi_aresetn;
  wire s_axi_arvalid;
  wire timeout_i;
  wire waddr_ready_sm1__0;

  LUT3 #(
    .INIT(8'h10)) 
    \DATA_PHASE_WDT.data_timeout_i_1 
       (.I0(\DATA_PHASE_WDT.data_timeout_reg_0 ),
        .I1(m_apb_pready),
        .I2(timeout_i),
        .O(\DATA_PHASE_WDT.data_timeout_reg ));
  LUT2 #(
    .INIT(4'hB)) 
    \icount_out[0]_i_1 
       (.I0(load_cntr__0),
        .I1(\icount_out_reg_n_0_[0] ),
        .O(\icount_out[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT4 #(
    .INIT(16'hEBBE)) 
    \icount_out[1]_i_1 
       (.I0(load_cntr__0),
        .I1(\icount_out_reg_n_0_[1] ),
        .I2(cntr_enable__2),
        .I3(\icount_out_reg_n_0_[0] ),
        .O(\icount_out[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT5 #(
    .INIT(32'hFABEEBFA)) 
    \icount_out[2]_i_1 
       (.I0(load_cntr__0),
        .I1(\icount_out_reg_n_0_[1] ),
        .I2(\icount_out_reg_n_0_[2] ),
        .I3(cntr_enable__2),
        .I4(\icount_out_reg_n_0_[0] ),
        .O(\icount_out[2]_i_1_n_0 ));
  LUT2 #(
    .INIT(4'hB)) 
    \icount_out[3]_i_1 
       (.I0(timeout_i),
        .I1(s_axi_aresetn),
        .O(\icount_out[3]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hABABAEFE)) 
    \icount_out[3]_i_2 
       (.I0(load_cntr__0),
        .I1(out[2]),
        .I2(out[1]),
        .I3(p_1_in__0),
        .I4(out[0]),
        .O(\icount_out[3]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFAFAFAEBBEFAFAFA)) 
    \icount_out[3]_i_3 
       (.I0(load_cntr__0),
        .I1(\icount_out_reg_n_0_[2] ),
        .I2(\icount_out_reg_n_0_[3] ),
        .I3(\icount_out_reg_n_0_[1] ),
        .I4(\icount_out_reg_n_0_[0] ),
        .I5(cntr_enable__2),
        .O(\icount_out[3]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h888888888888BBB8)) 
    \icount_out[3]_i_4 
       (.I0(\FSM_sequential_axi_wr_rd_cs_reg[1] ),
        .I1(out[0]),
        .I2(waddr_ready_sm1__0),
        .I3(s_axi_arvalid),
        .I4(out[1]),
        .I5(out[2]),
        .O(load_cntr__0));
  LUT6 #(
    .INIT(64'h011155550111AAAA)) 
    \icount_out[3]_i_5 
       (.I0(out[0]),
        .I1(\DATA_PHASE_WDT.data_timeout_reg_0 ),
        .I2(m_apb_pready),
        .I3(PENABLE_i_reg),
        .I4(out[1]),
        .I5(out[2]),
        .O(cntr_enable__2));
  LUT5 #(
    .INIT(32'h00000200)) 
    \icount_out[4]_i_1 
       (.I0(cntr_enable__2),
        .I1(\icount_out[4]_i_2_n_0 ),
        .I2(timeout_i),
        .I3(s_axi_aresetn),
        .I4(load_cntr__0),
        .O(\icount_out[4]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hF7FFFFEF)) 
    \icount_out[4]_i_2 
       (.I0(\icount_out_reg_n_0_[1] ),
        .I1(\icount_out_reg_n_0_[0] ),
        .I2(cntr_enable__2),
        .I3(\icount_out_reg_n_0_[2] ),
        .I4(\icount_out_reg_n_0_[3] ),
        .O(\icount_out[4]_i_2_n_0 ));
  FDRE \icount_out_reg[0] 
       (.C(s_axi_aclk),
        .CE(\icount_out[3]_i_2_n_0 ),
        .D(\icount_out[0]_i_1_n_0 ),
        .Q(\icount_out_reg_n_0_[0] ),
        .R(\icount_out[3]_i_1_n_0 ));
  FDRE \icount_out_reg[1] 
       (.C(s_axi_aclk),
        .CE(\icount_out[3]_i_2_n_0 ),
        .D(\icount_out[1]_i_1_n_0 ),
        .Q(\icount_out_reg_n_0_[1] ),
        .R(\icount_out[3]_i_1_n_0 ));
  FDRE \icount_out_reg[2] 
       (.C(s_axi_aclk),
        .CE(\icount_out[3]_i_2_n_0 ),
        .D(\icount_out[2]_i_1_n_0 ),
        .Q(\icount_out_reg_n_0_[2] ),
        .R(\icount_out[3]_i_1_n_0 ));
  FDRE \icount_out_reg[3] 
       (.C(s_axi_aclk),
        .CE(\icount_out[3]_i_2_n_0 ),
        .D(\icount_out[3]_i_3_n_0 ),
        .Q(\icount_out_reg_n_0_[3] ),
        .R(\icount_out[3]_i_1_n_0 ));
  FDRE \icount_out_reg[4] 
       (.C(s_axi_aclk),
        .CE(1'b1),
        .D(\icount_out[4]_i_1_n_0 ),
        .Q(timeout_i),
        .R(1'b0));
endmodule

(* ORIG_REF_NAME = "multiplexor" *) 
module axi_apb_bridge_0_multiplexor
   (m_apb_psel,
    p_0_in,
    PSEL_i,
    s_axi_aclk);
  output [0:0]m_apb_psel;
  input p_0_in;
  input PSEL_i;
  input s_axi_aclk;

  wire PSEL_i;
  wire [0:0]m_apb_psel;
  wire p_0_in;
  wire s_axi_aclk;

  FDRE \GEN_1_SELECT_SLAVE.M_APB_PSEL_i_reg[0] 
       (.C(s_axi_aclk),
        .CE(1'b1),
        .D(PSEL_i),
        .Q(m_apb_psel),
        .R(p_0_in));
endmodule
`ifndef GLBL
`define GLBL
`timescale  1 ps / 1 ps

module glbl ();

    parameter ROC_WIDTH = 100000;
    parameter TOC_WIDTH = 0;

//--------   STARTUP Globals --------------
    wire GSR;
    wire GTS;
    wire GWE;
    wire PRLD;
    tri1 p_up_tmp;
    tri (weak1, strong0) PLL_LOCKG = p_up_tmp;

    wire PROGB_GLBL;
    wire CCLKO_GLBL;
    wire FCSBO_GLBL;
    wire [3:0] DO_GLBL;
    wire [3:0] DI_GLBL;
   
    reg GSR_int;
    reg GTS_int;
    reg PRLD_int;

//--------   JTAG Globals --------------
    wire JTAG_TDO_GLBL;
    wire JTAG_TCK_GLBL;
    wire JTAG_TDI_GLBL;
    wire JTAG_TMS_GLBL;
    wire JTAG_TRST_GLBL;

    reg JTAG_CAPTURE_GLBL;
    reg JTAG_RESET_GLBL;
    reg JTAG_SHIFT_GLBL;
    reg JTAG_UPDATE_GLBL;
    reg JTAG_RUNTEST_GLBL;

    reg JTAG_SEL1_GLBL = 0;
    reg JTAG_SEL2_GLBL = 0 ;
    reg JTAG_SEL3_GLBL = 0;
    reg JTAG_SEL4_GLBL = 0;

    reg JTAG_USER_TDO1_GLBL = 1'bz;
    reg JTAG_USER_TDO2_GLBL = 1'bz;
    reg JTAG_USER_TDO3_GLBL = 1'bz;
    reg JTAG_USER_TDO4_GLBL = 1'bz;

    assign (strong1, weak0) GSR = GSR_int;
    assign (strong1, weak0) GTS = GTS_int;
    assign (weak1, weak0) PRLD = PRLD_int;

    initial begin
	GSR_int = 1'b1;
	PRLD_int = 1'b1;
	#(ROC_WIDTH)
	GSR_int = 1'b0;
	PRLD_int = 1'b0;
    end

    initial begin
	GTS_int = 1'b1;
	#(TOC_WIDTH)
	GTS_int = 1'b0;
    end

endmodule
`endif
